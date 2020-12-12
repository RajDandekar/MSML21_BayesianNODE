using DiffEqFlux, OrdinaryDiffEq, Flux, NNlib, MLDataUtils, Printf
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using CUDA
using Random: seed!
CUDA.allowscalar(false)

function loadmnist(batchsize = bs, train_split = 0.9)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata();
    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = stratifiedobs((x_data, y_data),
                                                         p = train_split)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(gpu.(collect.((x_train, y_train))); batchsize = batchsize,
                   shuffle = true),
        # Don't shuffle the test data
        DataLoader(gpu.(collect.((x_test, y_test))); batchsize = batchsize,
                   shuffle = false)
    )
end

# Main
const bs = 128
const train_split = 0.9
train_dataloader, test_dataloader = loadmnist(bs, train_split)

#down = Chain(flatten, Dense(784, 20, tanh)) |> gpu

nn = Chain(Dense(288, 64, relu),
           Dense(64, 64, relu),
           Dense(64, 288, relu)) |> gpu

nn2 = Chain(Dense(288, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 288, relu)) |> gpu

nn_ode2 = NeuralODE(nn2, (0.f0, 1.f0), Tsit5(),
        save_everystep = false,
        reltol = 1e-3, abstol = 1e-3,
        save_start = false) |> gpu
           
nn_ode = NeuralODE(nn, (0.f0, 1.f0), Tsit5(),
           save_everystep = false,
           reltol = 1e-3, abstol = 1e-3,
           save_start = false) |> gpu
           
nn_ode(randn(288, 1))

fc  = Chain(Dense(288, 10)) |> gpu

function DiffEqArray_to_Array(x)
    xarr = gpu(Array(x))
    return reshape(xarr, size(xarr)[1:2])
end

# Build our over-all model topology
model = Chain(Conv((3, 3), 1=>16, pad=(1,1), relu),
              x -> maxpool(x, (2,2)), Conv((3, 3), 16=>32, pad=(1,1), relu),
              x -> maxpool(x, (2,2)), Conv((3, 3), 32=>32, pad=(1,1), relu),
              x -> maxpool(x, (2,2)),
              x -> reshape(x, :, size(x, 4)),
              nn_ode,
              DiffEqArray_to_Array,
              nn_ode2,
              DiffEqArray_to_Array,
              fc) |> gpu;

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]

# We can see that we can compute the forward pass through the NN topology
# featuring an NNODE layer.
x_m = model(img)

classify(x) = argmax.(eachcol(x))

function accuracy(model, data; n_batches = 100)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        # Only evaluate accuracy for n_batches
        i > n_batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(model, train_dataloader)

loss(x, y) = logitcrossentropy(model(x), y)

# burn in loss


loss(img, lab)


#implementation of SGLD
function SGLD!(graddict, paramdict, a, b, γ, t)
    ε = a*(b + t)^-γ
    for p in paramdict
        ∇p = graddict[p]
        η = ε .* gpu(randn(size(p)))
        Δp = 0.5ε*∇p + η
        p .-= Δp
    end
end

function trainMNIST()
    seed!(1)
    weights = []
    iter = 0

    cb() = begin
        iter += 1
        # Monitor that the weights do infact update
        # Every 10 training iterations show accuracy
        if iter % 10 == 1
            train_accuracy = accuracy(model, train_dataloader) * 100
            test_accuracy = accuracy(model, test_dataloader;
                                     n_batches = length(test_dataloader)) * 100
            @printf("Iter: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
                    iter, train_accuracy, test_accuracy)
        end
    end

    Flux.@epochs 5 for (x, y) in train_dataloader
        g = gradient(() -> loss(x, y), params(model))
        SGLD!(g, params(model), .05, .5, 0.5, iter)
        cb()
        append!(weights, [deepcopy(params(model))])
    end
    weights
end
weights = trainMNIST()

weights[1] 

function createmodel(wts)
    conv1 = Conv(wts[1], wts[2], pad=(1,1), relu)
    conv2 = Conv(wts[3], wts[4], pad=(1,1), relu)
    conv3 = Conv(wts[5], wts[6], pad=(1,1), relu)
    mp = x -> maxpool(x, (2, 2))
    rs = x -> reshape(x, :, size(x, 4))
    node1 = x -> nn_ode(x, wts[7])
    node2 = x -> nn_ode2(x, wts[8])
    fc = Dense(288, 10, initW = (out, in) -> wts[9], initb = out -> wts[10]) |> gpu

    model = Chain(conv1, mp, conv2, mp, conv3, mp, rs,
                node1, DiffEqArray_to_Array, node2, DiffEqArray_to_Array, fc) |> gpu
end


#histogram and percentage calculations
using JLD, Zygote

#save("sampleparameters.jld", "weights", weights)
w = load("sampleparameters.jld", "weights")

sampls = w[1800:end]
m1 = createmodel(sampls[end])
Float64(classify(m1(img)) == lab)
m1(img)

ensemble_pred = Float64[]

test_im = test_dataloader.data[1][:, :, :, 25:25]
lab_test = test_dataloader.data[2][:,25:25]

preds_total = []

x_b = collect(test_dataloader)[10][1]
lab_b = classify(collect(test_dataloader)[10][2])
    
for s in sampls
    m = createmodel(s)
    pred = classify(m(x_b))
    append!(preds_total, [pred])
    println(size(pred))
end

preds_total

correct_pred = [lb .== lab_b for lb in preds_total]
sum(hcat(correct_pred...), dims = 2)

percent = sum(hcat(correct_pred...), dims = 2) ./ 311
_, minp =findmin(percent)
plot(percent)
percent[39]

using DelimitedFiles
?writedlm
writedlm("table", enumerate(percent), ',')
for x, t in enumerate(percent)

#Example of an image
using Plots, StatsPlots, StatsBase

test_im = x_b[:, :, :, 39:39]
test_label = lab_b[39]

img_test = heatmap(permutedims(test_im[:, :, 1, 1]), c = cgrad([:white, :black]), yflip = true, legend =:none)
savefig(img_test, "dubious7")
ensemble_pred = []

for s in sampls
    m = createmodel(s)
    pred = classify(m(test_im))
    append!(ensemble_pred, pred)
end


h1 = histogram(ensemble_pred .- 1, bins = 0.5:1:11, normalize = true, xticks = (1:10+0.5, 1:10), title = "Prediction (Ground truth = 7)", legend =:none)
savefig(h1, "hist7")


