cd("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE")
Pkg.activate(".")

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
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))


####DEFINE THE NEURAL ODE#####
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
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

l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)



function dldθ(θ)
    x,lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end

metric  = DiagEuclideanMetric(length(prob_neuralode.p))

h = Hamiltonian(metric, l, dldθ)


#CHECK HOW MUCH TIME IT TAKES TO RUN THIS FIND_GOOD_STEP_SIZE FUNCTION###
find_good_stepsize(h, Float64.(prob_neuralode.p))


integrator = Leapfrog(find_good_stepsize(h, Float64.(prob_neuralode.p)))


prop = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)

adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.45, prop.integrator))

samples, stats = @time sample(h, prop, Float64.(prob_neuralode.p), 500, adaptor, 1000; progress=true)


losses = map(x-> x[1],[loss_neuralode(samples[i]) for i in 1:length(samples)])

############################### PLOTAND SAVE LOSS ##################################
scatter(losses, ylabel = "Loss",  yscale= :log, label = "Architecture1: 1000 warmup, 500 sample")
savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/Extendedtime_SpiralODE_Loss_1000_500_Arch1.pdf")


################################PLOT RETRODICTED DATA ##########################
pl = scatter(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", xlabel = "t", title = "Spiral Neural ODE")
scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2")

for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(tsteps,resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps,resol[2,:], alpha=0.04, color = :blue, label = "")
end

idx = findmin(losses)[2]
prediction = predict_neuralode(samples[idx])

plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (-2.5, 3.5))


################################CONTOUR PLOTS##########################
pl = scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Spiral Neural ODE")

for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(resol[1,:],resol[2,:], alpha=0.04, color = :red, label = "")
end

plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (-2.5, 3))



################################FORECASTING PLOTS########################

u0 = [2.0; 0.0]
datasize_f = 50
tspan_f = (0.0, 1.2)
tsteps_f = range(tspan_f[1], tspan_f[2], length = datasize_f)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode_f = ODEProblem(trueODEfunc, u0, tspan_f)
ode_data_f = Array(solve(prob_trueode_f, Tsit5(), saveat = tsteps_f))


dudt2_f = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
prob_neuralode_f = NeuralODE(dudt2_f, tspan_f, Tsit5(), saveat = tsteps_f)

function predict_neuralode_f(p)
    Array(prob_neuralode_f(u0, p))
end

################################FORECASTING 1: PLOT RETRODICTED DATA ##########################
training_end = 1

pl = scatter(tsteps_f, ode_data_f[1,:], color = :red, label = "Data: Var1", xlabel = "t", title = "Spiral Neural ODE")
scatter!(tsteps_f, ode_data_f[2,:], color = :blue, label = "Data: Var2")

for k in 1:300
    resol = predict_neuralode_f(samples[100:end][rand(1:400)])
    plot!(tsteps_f[1:42], resol[1,:][1:42], alpha=0.04, color = :red, label = "")
    plot!(tsteps_f[1:42], resol[2,:][1:42], alpha=0.04, color = :blue, label = "")
    plot!(tsteps_f[42:end], resol[1,:][42:end], alpha=0.04, color = :purple, label = "")
    plot!(tsteps_f[42:end], resol[2,:][42:end], alpha=0.04, color = :purple, label = "")
end


idx = findmin(losses)[2]
prediction_f = predict_neuralode_f(samples[idx])

plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction", ylims = (-2.5, 3.5))
plot!([training_end-0.0001,training_end+0.0001],[-2.2,1.3],lw=3,color=:green,label="Training Data End", linestyle = :dash)

plot!(tsteps_f[42:end],prediction_f[1,:][42:end], color = :purple, w = 2, label = "")
plot!(tsteps_f[42:end],prediction_f[2,:][42:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2.5, 3.5))

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedSpiralODE_Retrodicted_500_1000_Arch1.pdf")


################################FORECASTING 2: CONTOUR PLOTS ##########################
pl = scatter(ode_data_f[1,:], ode_data_f[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Spiral Neural ODE")

for k in 1:300
    resol = predict_neuralode_f(samples[100:end][rand(1:400)])
    plot!(resol[1,:][1:42],resol[2,:][1:42], alpha=0.04, color = :red, label = "")
    plot!(resol[1,:][42:end],resol[2,:][42:end], alpha=0.1, color = :purple, label = "")

end

plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction", ylims = (-2.5, 3.5))
plot!(prediction_f[1,:][41:end], prediction_f[2,:][41:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2.5, 3.5))

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedSpiralODE_Contour_Retrodicted_500_1000_Arch1.pdf")

save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedSpiralNeuralODE_Architecture1_500_1000_0.45.jld", "losses",losses, "ode_data", ode_data, "samples", samples, "prediction", prediction, "stats", stats, "idx", idx, "tsteps", tsteps, "tsteps_f", tsteps_f )
