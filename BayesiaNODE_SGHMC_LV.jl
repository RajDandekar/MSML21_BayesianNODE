using Distributed
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Turing, Serialization, Plots
using JLD


u0 = Float32[1., 1.];
p = [1.5, 1., 3., 1.];
tspan = (0.0f0, 4.5f0);
tsteps = tspan[1]:0.1:tspan[2];

function lv(u, p, t)
    x, y = u
    α, β, γ, δ = p
    dx = α*x - β*x*y
    dy = δ*x*y - γ*y
    du = [dx, dy]
end


trueodeprob = ODEProblem(lv, u0, tspan, p);
ode_data = Array(solve(trueodeprob, Tsit5(), saveat = tsteps));
y_train = ode_data[:, 1:35];

dudt2 = FastChain(FastDense(2, 10, tanh), FastDense(10, 2));

prob_node = NeuralODE(dudt2, (0., 4.5), Tsit5(), saveat = tsteps); #neural ode
train_prob = NeuralODE(dudt2, (0., 3.5), Tsit5(), saveat = tsteps[1:35]);


function predict_node(p)  # predict with given params
    Array(train_prob(u0, p))
end

function loss(p)  # loss function to minimize
    Float64(sum(abs2, y_train .- predict_node(p)))
end


####### Perform inference


### Fit neural ode to the data
@everywhere @model function fit_node(data)
    σ ~ InverseGamma(2, 3)
    p ~ MvNormal(pmin_lv, 0.4)

    # Calculate predictions for the inputs given the params.
    predicted = predict_node(p)

    # observe each prediction.
    for i = 1:size(predicted,2)
        data[:,i] ~ MvNormal(predicted[:,i], σ)
    end
end

@everywhere model = fit_node(y_train);
function perform_inference(lr, alpha, samplesize, num_chains)
    alg = SGHMC(learning_rate=lr, momentum_decay=alpha)
    chain = sample(model, alg, MCMCThreads(), samplesize, num_chains, init_theta=zero(pmin_lv), progress=true);
    return chain
end

function map_loss(chain)
    chain_array = Array(chain)
    k = size(chain_array,1)
    losses = loss.([chain_array[i,:] for i in 1:k])
    return losses
end

# init at map point
using JLD
pinit = initial_params(dudt2);
opt = DiffEqFlux.sciml_train(loss, pinit, ADAM(0.05), maxiters = 1500)
opt2 = DiffEqFlux.sciml_train(loss, opt.minimizer, LBFGS(), allow_f_increases = true)
pmin = opt2.minimizer;
save("pmin_lv.jld", "pmin_lv", pmin)

using JLD
pmin = load("pmin_lv.jld")
pmin = pmin["pmin_lv"]
pmin_lv = pmin;


function plot_chain(chain, losses)
    pl = plot()
    chain_array = Array(chain)
    len = size(chain_array,1)

    training_end = 3.5

    scatter!(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", title = "Lotka Volterra Neural ODE")
    scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2")

    for k in 1:300
        resol = prob_node(u0, chain_array[rand(25:len), :])
        plot!(tsteps[1:36], resol[1,1:36], alpha=0.04, color=:red, label = "")
        plot!(tsteps[1:36], resol[2,1:36], alpha=0.04, color=:blue, label = "")
        plot!(tsteps[36:45], resol[1,36:45], alpha=0.04, color=:purple, label = "")
        plot!(tsteps[36:45], resol[2,36:45], alpha=0.04, color=:purple, label = "")
    end

    idx = findmin(losses)[2]
    prediction = prob_node(u0, chain_array[idx, :])
    plot!(tsteps, prediction[1,:], color=:black, w=2, label = "")
    plot!(tsteps, prediction[2,:], color=:black, w=2, label = "Training: Best fit prediction")
    plot!(tsteps[36:end], prediction[1,:][36:end], color = :purple, w = 2, label = "")
    plot!(tsteps[36:end], prediction[2,:][36:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-1.5, 10))

    display(plot!([training_end-0.0001,training_end+0.0001],[-1,5],lw=3,color=:green,label="Training Data End", linestyle = :dash))

    ############## CONTOUR PLOTS #######################
    pl2 = scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Lotka Volterra Neural ODE")

    for k in 1:300
        resol = prob_node(u0, chain_array[rand(100:len), :])
        plot!(resol[1,:][1:36],resol[2,:][1:36], alpha=0.04, color = :red, label = "")
        plot!(resol[1,:][36:end],resol[2,:][36:end], alpha=0.1, color = :purple, label = "")
    end

    plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction")
    display(plot!(prediction[1,:][36:end], prediction[2,:][36:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2, 7), xlims = (-0.5, 7.5)))


    return pl, pl2;
end

## ---------------------------------------------------


lr = 1.5e-6; md = 0.07; samples = 2500

num_chains = 7

chain = perform_inference(lr, md, samples, num_chains);
for i in 1:num_chains
    losses = map_loss(chain[:,:,i])
    pl = plot(1:samples, losses); display(pl)
    savefig(pl, string(lr, "_", md, "_", samples, "_", "chain_", i, "_losses", ".png"))
    pl_ch, pl2 = plot_chain(chain[:,:,i], losses)
    savefig(pl_ch, string(lr, "_", md, "_", samples, "_", "chain_", i, "_predictions", ".png"))
    savefig(pl2, string(lr, "_", md, "_", samples, "_", "chain_", i, "_contour", ".png"))

end
