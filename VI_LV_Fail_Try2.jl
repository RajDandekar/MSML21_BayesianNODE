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


Loss_all = rand(500)

for i = 1:length(Loss_all)
    Loss_all[i] = loss_neuralode(post[:,i])[1]
end

density(Loss_all)

samples_reduced = post[1:5, :]

samples_reshape = reshape(samples_reduced, (500, 5, 1))

Chain_LV = Chains(samples_reshape)

plot(Chain_LV)

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/Spiral_DiagnosisPlot1.pdf")

autocorplot(Chain_LV)

savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE/Spiral_DiagnosisPlot2.pdf")
