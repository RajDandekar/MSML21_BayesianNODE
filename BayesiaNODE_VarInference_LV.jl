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


####DEFINE THE NEURAL ODE#####
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

callback = function (p, l, pred)
  display(l)
  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.01), cb = callback,
                                          maxiters = 10000)

#result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode, result_neuralode.minimizer,LBFGS(),cb = callback,maxiters = 10000)

p_opt = result_neuralode.minimizer

l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ)) - sum(θ .* θ)

#l(θ) = -sum(abs2, ode_data .- predict_neuralode(θ))


function dldθ(θ)
    x,lambda = Flux.Zygote.pullback(l,θ)
    grad = first(lambda(1))
    return x, grad
end

metric  = DiagEuclideanMetric(length(prob_neuralode.p))

h = Hamiltonian(metric, l, dldθ)

advi = ADVI(10, 1000)

getq(θ) = TuringDiagMvNormal(θ[1:length(prob_neuralode.p)], (0.01*θ[1:length(prob_neuralode.p)].^2))

q = vi(h.ℓπ, advi, getq, p_opt)

post = rand(q, 10000)

#=
p_mean = rand(length(p_opt))
for i = 1:length(p_opt)
    p_mean[i] = mean(post[i, :])
end
=#


Loss_all = rand(10000)

for i = 1:length(Loss_all)
    Loss_all[i] = loss_neuralode(post[:,i])[1]
end

density(Loss_all)

savefig("C:/Users/16174/Desktop/Julia Lab/MSML2021/Loss_LV_VI_fail.pdf")

using JLD
save("C:/Users/16174/Desktop/Julia Lab/MSML2021/LV_VI_fail.jld", "q", q, "post", post)

##################LOAD AND REPLOT STUFF#############
D = load("C:/Users/16174/Desktop/Julia Lab/MSML2021/LV_VI_fail.jld")
q = D["q"]

post = rand(q, 500)

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
    resol = predict_neuralode_f(post[:, rand(1:400)])
    plot!(tsteps_f[1:36], resol[1,:][1:36], alpha=0.04, color = :red, label = "")
    plot!(tsteps_f[1:36], resol[2,:][1:36], alpha=0.04, color = :blue, label = "")
    plot!(tsteps_f[36:end], resol[1,:][36:end], alpha=0.04, color = :purple, label = "")
    plot!(tsteps_f[36:end], resol[2,:][36:end], alpha=0.04, color = :purple, label = "")
end

display(pl)

plot!([training_end-0.0001,training_end+0.0001],[-1,5],lw=3,color=:green,label="Training Data End", linestyle = :dash)

savefig("C:/Users/16174/Desktop/Julia Lab/MSML2021/LV_VI_fail_plot1.pdf")

###########CONTOUR FORECASTING PLOT###################
pl = scatter(ode_data_f[1,:], ode_data_f[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Lotka Volterra Neural ODE")

for k in 1:300
    resol = predict_neuralode_f(post[:, rand(1:400)])
    plot!(resol[1,:][1:36],resol[2,:][1:36], alpha=0.04, color = :red, label = "")
    plot!(resol[1,:][36:end],resol[2,:][36:end], alpha=0.1, color = :purple, label = "")
end

display(pl)

savefig("C:/Users/16174/Desktop/Julia Lab/MSML2021/LV_VI_fail_plot2.pdf")


