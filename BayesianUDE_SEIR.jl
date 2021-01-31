#############SHOWING THAT SEIR ALSO WORKS WITH SGLD APPROACH!
#########FOR DIFF LAMBDA AND FOR ALL ITERATIONS, SAME TERMS RECOVERED.
#####NEED TO CAPTURE AICC FOR DIFF LAMBDA AND 100 ITERATIONS#######

cd("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural ODE")
Pkg.activate(".")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, AdvancedHMC, Serialization
using JLD, StatsPlots
using Random

function SEIR!(du, u, p, t)
    S, E, I, R = u
    β, γ, σ = p

    du[1] = -β*S*I
    du[2] = β*S*I - σ*E
    du[3] = σ*E - γ*I
    du[4] = γ*I

end

p = [0.85, 0.2, 0.1]

S0 = 1.
u0 = [S0*0.99,S0*0.01, S0*0.01, 0.]
tspan = (0., 60.)

prob_ode = ODEProblem(SEIR!, u0, tspan, p)

sol_ode = solve(prob_ode, Tsit5(), saveat = 1.0)

ode_data = hcat([sol_ode[:,i]  for i in 1:size(sol_ode,2)]...)



######DEFINE THE UDE##########
global β, γ, σ1 = [0.85, 0.2, 0.1]


function SEIR_ude!(du, u, p, t)
  global β, γ, σ1
  S, E, I, R = u

  ucollect = [u[1]; u[2]; u[3]; u[4]]
  NN1 = abs(re(p[1:121])(ucollect)[1])

  du[1] = -β*S*I
  du[2] = β*S*I - 0.01*NN1
  du[3] = σ1*E - γ*I
  du[4] = γ*I
end

# Initial condition
u0 = [S0*0.99,S0*0.01, S0*0.01, 0.]
tspan = (0., 60.)


ann = Chain(Dense(4,20,relu), Dense(20,1))
p1,re = Flux.destructure(ann)

# Setup the ODE problem, then solve
prob_ude = ODEProblem(SEIR_ude!, u0, tspan, p1)

#Array(concrete_solve(prob_ude,Tsit5(),u0,p1,saveat = 2.0))

function train_neuralsir(steps, a, b, γ)
    θ = p1

    predict(p) = Array(concrete_solve(prob_ude,Tsit5(),u0,p,saveat = 1.0))

	loss(p) = sum(abs2, ode_data .- predict(p))

	trainlosses = [loss(p1); zeros(steps)]

	weights = [p1'; zeros(steps, length(p1))]

    for t in 1:steps
	##########DEFINE########################
	beta = 0.9;
	λ =1e-8;
	precond = zeros(length(θ))

	################GRADIENT CALCULATIONS
	 x,lambda = Flux.Zygote.pullback(loss,θ)
	 ∇L  = first(lambda(1))
	 #∇L= Flux.Zygote.gradient(loss, θ)[1]
	 #ϵ = 0.0001
	 ϵ = a*(b + t)^-γ

	###############PRECONDITIONING#####################
	 if t == 1
	   precond[:] = ∇L.*∇L
     else
	   precond *= beta
	   precond += (1-beta)*(∇L .*∇L)
     end
	 #m = λ .+ sqrt.(precond/((1-(beta)^t)))
	 m = λ .+ sqrt.(precond)

	 ###############DESCENT###############################
	 for i in 1:length(∇L)
		 noise = ϵ*randn()
		 #θ[i] = θ[i] - (0.5*ϵ*∇L[i] +  noise)
		 θ[i] = θ[i] - (0.5*ϵ*∇L[i]/m[i] +  noise)
	 end

	#################BOOKKEEPING############################
	weights[t+1, :] = p1
	trainlosses[t+1] = loss(p1)
	println(loss(p1))
	end
    print("Final loss is $(trainlosses[end])")

    trainlosses, weights
end

#results =  train_neuralsir(6000, 0.0000006)

#results =  train_neuralsir(6000, 0.0000006, 10, 1e-6, 0.9)

results =  train_neuralsir(20000, 0.0001, 0.85, 0.01)


trainlosses, parameters = results;

#save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_SEIR_morelayers_NeuralUDE.jld", "parameters",parameters, "trainlosses", trainlosses, "ode_data", ode_data)

println(trainlosses[end])
p = plot(trainlosses, scale =:log10)
savefig(p, "lossesSGLD_LV_UDE") #loss is around ~7, no visible sampling phase, more iters needed

################################PLOT RETRODICTED DATA ##########################
function predict_neuralode(p)
    Array(concrete_solve(prob_ude,Tsit5(),u0,p,saveat = 1.0))
end

tsteps = 0.0:1.0:tspan[end]

pl = Plots.scatter(tsteps, ode_data[1,:], color = :red, label = "Data: Susceptible", xlabel = "t", title = "SEIR Example")
Plots.scatter!(tsteps, ode_data[2,:], color = :orange, label = "Data: Exposed")
Plots.scatter!(tsteps, ode_data[3,:], color = :green, label = "Data: Infected")
Plots.scatter!(tsteps, ode_data[4,:], color = :blue, label = "Data: Recovered")

for k in 1:500
    resol = predict_neuralode(parameters[end-rand(1:600), :])
    plot!(tsteps,resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps,resol[2,:], alpha=0.04, color = :orange, label = "")
	plot!(tsteps,resol[3,:], alpha=0.04, color = :green, label = "")
	plot!(tsteps,resol[4,:], alpha=0.04, color = :blue, label = "")
end

idx = findmin(trainlosses[end-400:end])[2]
prediction = predict_neuralode(parameters[end- 400 + idx, :])

plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Best fit prediction")
plot!(tsteps,prediction[3,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[4,:], color = :black, w = 2, label = "")

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_SEIR_Plots1_T1.pdf")

########################PLOTS OF THE RECOVERED TERM: A#####################
Actual_Term = σ1*ode_data[2,:]

UDE_SGLD= zeros(Float64, length(Actual_Term), 1)

function UDE_term(UDE_SGLD, idx)
	prediction = predict_neuralode(parameters[end-idx, :])
	s_sol = prediction[1, :]
	e_sol = prediction[2, :]
	i_sol = prediction[3, :]
	r_sol = prediction[4, :]

	for i = 1:length(s_sol)
	  UDE_SGLD[i] = 0.01*abs(re(parameters[end-idx, :][1:121])([s_sol[i],e_sol[i], i_sol[i], r_sol[i]])[1])
	end

	return s_sol, e_sol, i_sol, r_sol, UDE_SGLD
end

pl = Plots.scatter(tsteps, Actual_Term, color = :red, label = "Data",  xlabel = "t", ylabel = "E(t)", title = "SEIR UDE Example")

for k in 1:300
    s_sol, e_sol, i_sol, r_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
    plot!(tsteps,UDE_SGLD[1:end], alpha=0.04, color = :red, label = "")
end


idx = findmin(trainlosses[end-400:end])[2]
prediction = predict_neuralode(parameters[end- 400 + idx, :])
s_sol = prediction[1, :]
e_sol = prediction[2, :]
i_sol = prediction[3, :]
r_sol = prediction[4, :]

for i = 1:length(s_sol)
	UDE_SGLD[i] = 0.01*abs(re(parameters[end-idx, :][1:121])([s_sol[i],e_sol[i], i_sol[i], r_sol[i]])[1])
end

plot!(tsteps,UDE_SGLD[1:end], color = :black, w = 2, label = "Best fit prediction", legend = :bottomright)

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_SEIR_Plots2_T1.pdf")


save("C:/Users/16174/OneDrive/Desktop/Julia Lab/ICML21/SGLD_ADAM_SEIR_NeuralUDE_T_60.jld", "parameters",parameters, "trainlosses", trainlosses, "ode_data", ode_data)


D = load("C:/Users/16174/OneDrive/Desktop/Julia Lab/ICML21/SGLD_ADAM_SEIR_NeuralUDE_T_60.jld")
parameters = D["parameters"]
trainlosses = D["trainlosses"]


### Universal ODE Part 2: SInDy to Equations
using DataDrivenDiffEq
using ModelingToolkit
# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[]
for i ∈ 0:1, j ∈ 0:1
    push!(polys, u[1]^i * u[2]^j )
end


h = [unique(polys)...]
basis = Basis(h, u)

###################After this point, you have to use the following steps#########

# Step1: Find the optimal λ:

###Generally the optimal λ is the one with lowest positive aicc score.
#####Note that the j loop times out after 100 iteration, I have not fully optimized it yet. I am currently running it till 20 in this code
####Note that ultimately we want the least amount of active terms.

λ = [0.005, 0.05, 0.1]
AICC_Score = zeros(1,3)

for i in 3
opt = STRRidge(λ[i])

aicc_store = 0

for j in 1:100
s_sol, e_sol, i_sol, r_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
Ψ = SInDy(vcat(e_sol[1:end]', i_sol[1:end]'), UDE_SGLD[1:end], basis, opt = opt, maxiter = 10000, normalize = true) # Suceed

aicc_store += Ψ.aicc[1]
end

AICC_Score[i] = aicc_store/100

end

###0.005 - 22.1172
###0.05 - 22.118
###0.1 - 22.1184
###0.5 - 0 terms recovered


####Step 2:  Note down terms for all runs. In this case, I get the same terms each time######
for i in 1:100
s_sol, e_sol, i_sol, r_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
Ψ = SInDy(vcat(e_sol[1:end]', i_sol[1:end]'), UDE_SGLD[1:end], basis, opt = opt, maxiter = 10000, normalize = true) # Suceed

print_equations(Ψ,  show_parameter = true)
end

####Step3: Note down probability of occurence of different terms based on the result of the 100 runs########
