#cd("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE")
#Pkg.activate(".")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC
using JLD, StatsPlots

u0 = [2.0; 0.0]
datasize = 40
tspan = (0.0, 1)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
mean_ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
ode_data = mean_ode_data .+ 0.1 .* randn(size(mean_ode_data)..., 30)

####DEFINE THE NEURAL ODE#####
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, relu),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

function l(θ)
    lp = logpdf(MvNormal(zeros(length(θ) - 1), θ[end]), θ[1:end-1])
    ll = -sum(abs2, ode_data .- predict_neuralode(θ[1:end-1]))
    return lp + ll
end
function dldθ(θ)
    x, lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end

init = [Float64.(prob_neuralode.p); 1.0]
opt = DiffEqFlux.sciml_train(x -> -l(x), init, ADAM(0.05), maxiters = 1500)
pmin = opt.minimizer;

metric  = DiagEuclideanMetric(length(pmin))
h = Hamiltonian(metric, l, dldθ)
integrator = Leapfrog(find_good_stepsize(h, pmin))
prop = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator, 5)

adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.5, prop.integrator))

samples, stats = sample(h, prop, pmin, 500, adaptor, 500; progress=true)

losses = map(x-> x[1],[loss_neuralode(samples[i][1:end-1]) for i in 1:length(samples)])

############################### PLOTAND SAVE LOSS ##################################
scatter(losses, ylabel = "Loss",  yscale= :log, label = "Architecture1: 1000 warmup, 500 sample")
savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/Extendedtime_SpiralODE_Loss_1000_500_Arch1.pdf")


################################PLOT RETRODICTED DATA ##########################
pl = scatter(tsteps, mean_ode_data[1,:], color = :red, label = "Data: Var1", title = "Spiral Neural ODE")
scatter!(tsteps, mean_ode_data[2,:], color = :blue, label = "Data: Var2", xlabel = "t", ylims = (0, 10))
for _ in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)][1:end-1])
    plot!(tsteps, resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps, resol[2,:], alpha=0.04, color = :blue, label = "")
end
resol = predict_neuralode(samples[100:end][rand(1:400)][1:end-1])
plot!(tsteps, resol[1,:], alpha=0.04, color = :red, label = "")
plot!(tsteps, resol[2,:], alpha=0.04, color = :blue, label = "")

idx = findmin(losses)[2]
prediction = predict_neuralode(samples[idx][1:end-1])

plot!(tsteps, prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps, prediction[2,:], color = :black, w = 2, label = "Best fit prediction")


################################CONTOUR PLOTS##########################
pl = scatter(
    mean_ode_data[1,:],
    mean_ode_data[2,:],
    color = :blue, label = "Data",  xlabel = "Var1",
    ylabel = "Var2", title = "Spiral Neural ODE",
    legend = (0.85, 0.95), legendfontsize = 5,
)
for k in 1:size(ode_data, 3)
    scatter!(
        ode_data[1,:,k],
        ode_data[2,:,k],
        color = :blue, label = "",
    )
end

for k1 in 301:500
    σ = samples[k1][end]
    resol = predict_neuralode(samples[k1][1:end-1])
    for k2 in 1:10
        _resol = resol .+ σ .* randn.()
        label = ""
        plot!(_resol[1,:], _resol[2,:], alpha=0.02, color = :red, label = label)
    end
end

plot!(prediction[1,:], prediction[2,:], color = :red, w = 2, label = "Simulated data")
plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction")


################################FORECASTING PLOTS########################

u0 = [2.0; 0.0]
datasize_f = 50
tspan_f = (0.0, 1.2)
tsteps_f = [tsteps; range(tspan[2], tspan_f[2], length = datasize_f - datasize)]

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode_f = ODEProblem(trueODEfunc, u0, tspan_f)
mean_ode_data_f = Array(solve(prob_trueode_f, Tsit5(), saveat = tsteps_f))
ode_data_f = cat(ode_data, mean_ode_data_f[:, datasize+1:end] .+ 0.1 .* randn(size(mean_ode_data_f, 1), size(mean_ode_data_f, 2) - datasize, 30), dims = 2)

prob_neuralode_f = NeuralODE(dudt2, tspan_f, Tsit5(), saveat = tsteps_f)

function predict_neuralode_f(p)
    Array(prob_neuralode_f(u0, p))
end

idx = findmin(losses)[2]
prediction_f = predict_neuralode_f(samples[idx])

training_end = 1
pl = scatter(
    mean_ode_data_f[1,:],
    mean_ode_data_f[2,:],
    color = :blue, label = "Data",  xlabel = "Var1",
    ylabel = "Var2", title = "Spiral Neural ODE",
    legend = (0.8, 0.95), legendfontsize = 5,
)

for k in 1:size(ode_data_f, 3)
    scatter!(
        ode_data_f[1,:,k],
        ode_data_f[2,:,k],
        color = :blue, label = "",
    )
end

for k1 in 301:500
    σ = samples[k1][end]
    resol = predict_neuralode_f(samples[k1][1:end-1])
    for k2 in 1:10
        _resol = resol .+ σ .* randn.()
        label = ""
        plot!(_resol[1,:][1:datasize], _resol[2,:][1:datasize], alpha=0.02, color = :red, label = label)
        plot!(_resol[1,:][datasize+1:end], _resol[2,:][datasize+1:end], alpha=0.02, color = :green, label = label)
    end
end

plot!(prediction_f[1,1:datasize], prediction_f[2,1:datasize], color = :red, w = 2, label = "Training: simulated data")
plot!(prediction_f[1,datasize+1:end], prediction_f[2,datasize+1:end], color = :green, w = 2, label = "Forecasting: simulated data")
plot!(prediction_f[1,:], prediction_f[2,:], color = :black, w = 2, label = "Best fit prediction")

savefig("ExtendedSpiralODE_Contour_Retrodicted_500_500_Arch1.pdf")

save("ExtendedSpiralNeuralODE_Architecture1_500_500_0.50.jld", "losses",losses, "ode_data", ode_data, "samples", samples, "prediction", prediction, "stats", stats, "idx", idx, "tsteps", tsteps, "tsteps_f", tsteps_f )
