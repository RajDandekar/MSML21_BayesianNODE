using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 3.5)
tsteps = 0.0:0.1:3.5

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5(), saveat = tsteps)
ode_data = hcat([sol_ode[:,i] for i in 1:size(sol_ode,2)]...)


dudt2 = FastChain(FastDense(2, 20, tanh),
                  FastDense(20, 2))
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
integrator = Leapfrog(find_good_stepsize(h, Float64.(prob_neuralode.p)))
prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)

adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.45, prop.integrator))

samples, stats = sample(h, prop, Float64.(prob_neuralode.p), 500, adaptor, 2000; progress=true)


losses = map(x-> x[1],[loss_neuralode(samples[i]) for i in 1:length(samples)])

############################### PLOT LOSS ##################################
scatter(losses, ylabel = "Loss",  label = "Architecture1: 2000 warmup, 500 sample")

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedLV_Loss_500_2000_Arch1.pdf")

################################PLOT RETRODICTED DATA ##########################
pl = scatter(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", title = "Lotka Volterra Neural ODE")
scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2", xlabel = "t", ylims = (0, 10))

for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(tsteps, resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps, resol[2,:], alpha=0.04, color = :blue, label = "")
end

idx = findmin(losses)[2]
prediction = predict_neuralode(samples[idx])

plot!(tsteps, prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps, prediction[2,:], color = :black, w = 2, label = "Best fit prediction")


################################CONTOUR PLOTS##########################
pl = scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Lotka Volterra Neural ODE")

for k in 1:300
    resol = predict_neuralode(samples[100:end][rand(1:400)])
    plot!(resol[1,:],resol[2,:], alpha=0.04, color = :red, label = "")
end

plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction")


############################FORECASTING###################
function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0_f = [1.0, 1.0]

# Simulation interval and intermediary points
tspan_f = (0.0, 4.5)
tsteps_f = 0.0:0.1:4.5

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode_f = ODEProblem(lotka_volterra!, u0_f, tspan_f, p)
sol_ode_f = solve(prob_ode_f, Tsit5(), saveat = tsteps_f)
ode_data_f = hcat([sol_ode_f[:,i] for i in 1:size(sol_ode_f,2)]...)

dudt2_f = FastChain(FastDense(2, 20, tanh),
                  FastDense(20, 2))
prob_neuralode_f = NeuralODE(dudt2_f, tspan_f, Tsit5(), saveat = tsteps_f)

function predict_neuralode_f(p)
    Array(prob_neuralode_f(u0_f, p))
end

training_end = 3.5

pl = scatter(tsteps_f, ode_data_f[1,:], color = :red, label = "Data: Var1", xlabel = "t", title = "Lotka Volterra Neural ODE")
scatter!(tsteps_f, ode_data_f[2,:], color = :blue, label = "Data: Var2")

for k in 1:300
    resol = predict_neuralode_f(samples[100:end][rand(1:400)])
    plot!(tsteps_f[1:36], resol[1,:][1:36], alpha=0.04, color = :red, label = "")
    plot!(tsteps_f[1:36], resol[2,:][1:36], alpha=0.04, color = :blue, label = "")
    plot!(tsteps_f[36:end], resol[1,:][36:end], alpha=0.04, color = :purple, label = "")
    plot!(tsteps_f[36:end], resol[2,:][36:end], alpha=0.04, color = :purple, label = "")
end


idx = findmin(losses)[2]
prediction_f = predict_neuralode_f(samples[idx])

plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction")

plot!([training_end-0.0001,training_end+0.0001],[-1,5],lw=3,color=:green,label="Training Data End", linestyle = :dash)

plot!(tsteps_f[36:end],prediction_f[1,:][36:end], color = :purple, w = 2, label = "")
plot!(tsteps_f[36:end],prediction_f[2,:][36:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-1.5, 10))


savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedLV_Retrodicted_500_2000_Arch2.pdf")

################################FORECASTING 2: CONTOUR PLOTS ##########################
pl = scatter(ode_data_f[1,:], ode_data_f[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Lotka Volterra Neural ODE")

for k in 1:300
    resol = predict_neuralode_f(samples[100:end][rand(1:400)])
    plot!(resol[1,:][1:36],resol[2,:][1:36], alpha=0.04, color = :red, label = "")
    plot!(resol[1,:][36:end],resol[2,:][36:end], alpha=0.1, color = :purple, label = "")
end

plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Training: Best fit prediction")
plot!(prediction_f[1,:][36:end], prediction_f[2,:][36:end], color = :purple, w = 2, label = "Forecasting: Best fit prediction", ylims = (-2, 7), xlims = (-0.5, 7))

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedLV_Contour_Retrodicted_500_2000_Arch2.pdf")

save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/ExtendedLVNeuralODE_Architecture2_500_2000_0.45.jld", "losses",losses, "ode_data", ode_data, "samples", samples, "prediction", prediction, "stats", stats, "idx", idx, "tsteps", tsteps, "tsteps_f", tsteps_f )
