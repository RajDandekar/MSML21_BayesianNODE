cd("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural ODE")
Pkg.activate(".")

using DiffEqFlux, DifferentialEquations, Flux, Optim, Plots, AdvancedHMC, Serialization
using JLD, StatsPlots
using Random

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 2.0)
tsteps = 0.0:0.1:2.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)

sol_ode = solve(prob_ode, Tsit5(), saveat = tsteps)

ode_data = hcat([sol_ode[:,i] for i in 1:size(sol_ode,2)]...)

global α, β, δ, γ = [1.5, 1.0, 3.0, 1.0]

function lotka_volterra_ude!(du, u, p, t)
  global α, β, δ, γ
  ucollect = [u[1]; u[2]]
  NN1 = abs(re(p[1:81])(ucollect)[1])
  x, y = u
  du[1] = α*x -NN1
  du[2] = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 2.0)
tsteps = 0.0:0.1:2.0

ann = Chain(Dense(2,20,tanh),  Dense(20,1))
p1,re = Flux.destructure(ann)

# Setup the ODE problem, then solve
prob_ude = ODEProblem(lotka_volterra_ude!, u0, tspan, p1)



function train_neuralsir(steps, a, b, γ)
    θ = p1

    predict(p) = Array(concrete_solve(prob_ude,Tsit5(),u0,p,saveat = tsteps))

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

results =  train_neuralsir(20000, 0.0001, 0.085, 0.001)

trainlosses, parameters = results;

println(trainlosses[end])
p = plot(trainlosses, scale =:log10)
savefig(p, "lossesSGLD_LV_UDE") #loss is around ~7, no visible sampling phase, more iters needed

################################PLOT RETRODICTED DATA ##########################
function predict_neuralode(p)
    Array(concrete_solve(prob_ude,Tsit5(),u0,p,saveat = tsteps))
end

pl = Plots.scatter(tsteps, ode_data[1,:], color = :red, label = "Data: Var1", xlabel = "t", title = "Spiral Neural ODE")
Plots.scatter!(tsteps, ode_data[2,:], color = :blue, label = "Data: Var2")

for k in 1:500
    resol = predict_neuralode(parameters[end-rand(1:600), :])
    plot!(tsteps,resol[1,:], alpha=0.04, color = :red, label = "")
    plot!(tsteps,resol[2,:], alpha=0.04, color = :blue, label = "")
end

idx = findmin(trainlosses[end-400:end])[2]
prediction = predict_neuralode(parameters[end- 400 + idx, :])

plot!(tsteps,prediction[1,:], color = :black, w = 2, label = "")
plot!(tsteps,prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (0, 9))

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_Plots1.pdf")

################################CONTOUR PLOTS##########################
pl = Plots.scatter(ode_data[1,:], ode_data[2,:], color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Spiral Neural ODE")

for k in 1:300
    resol = predict_neuralode(parameters[end-rand(1:300), :])
    plot!(resol[1,:],resol[2,:], alpha=0.04, color = :red, label = "")
end

plot!(prediction[1,:], prediction[2,:], color = :black, w = 2, label = "Best fit prediction", ylims = (0, 2.5) )

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_Plots2.pdf")


########################PLOTS OF THE RECOVERED TERM: A#####################
Actual_Term = 1* ode_data[1,:] .* ode_data[2,:]

UDE_SGLD= zeros(Float64, length(Actual_Term), 1)

function UDE_term(UDE_SGLD, idx)
	prediction = predict_neuralode(parameters[end-idx, :])
	x_sol = prediction[1, :]
	y_sol = prediction[2, :]
	for i = 1:length(x_sol)
	  UDE_SGLD[i] = abs(re(parameters[end-idx, :][1:81])([x_sol[i],y_sol[i]])[1])
	end
	return x_sol, y_sol, UDE_SGLD
end

pl = Plots.scatter(tsteps, Actual_Term, color = :red, label = "Data",  xlabel = "Var1", ylabel = "Var2", title = "Lotka Volterra UDE")

for k in 1:300
    x_sol, y_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
    plot!(tsteps,UDE_SGLD[1:end], alpha=0.04, color = :red, label = "")
end


idx = findmin(trainlosses[end-400:end])[2]
prediction = predict_neuralode(parameters[end- 400 + idx, :])
x_sol = prediction[1, :]
y_sol = prediction[2, :]
for i = 1:length(x_sol)
  UDE_SGLD[i] = abs(re(parameters[end- 400 + idx, :][1:81])([x_sol[i],y_sol[i]])[1])
end

plot!(tsteps,UDE_SGLD[1:end], color = :black, w = 2, label = "Best fit prediction")

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_Plots3.pdf")

########################PLOTS OF THE RECOVERED TERM: B#####################
#=
plotlyjs()

pl = scatter3d(x = ode_data[1,:], y = ode_data[2,:], z = Actual_Term, mode="markers",opacity=0.8,
					marker_size=6, marker_line_width=0.5,
					marker_line_color="rgba(217, 217, 217, 0.14)")
plot(pl)
=#
gr()

Plots.scatter(ode_data[1,:], ode_data[2,:], Actual_Term, xlabel = "x", ylabel = "y", title = "Lotka Volterra UDE")
for k in 1:300
    x_sol, y_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
    plot!(x_sol, y_sol,UDE_SGLD[1:end], alpha=0.04, color = :red, label = "")
end

idx = findmin(trainlosses[end-400:end])[2]
prediction = predict_neuralode(parameters[end- 400 + idx, :])
x_sol = prediction[1, :]
y_sol = prediction[2, :]
for i = 1:length(x_sol)
  UDE_SGLD[i] = abs(re(parameters[end- 400 + idx, :][1:81])([x_sol[i],y_sol[i]])[1])
end

plot!(x_sol, y_sol, UDE_SGLD[1:end], color = :black, w =2,   label = "Best fit prediction", ylims = (0, 15))

Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_Plots4.pdf")

save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_NeuralUDE.jld", "parameters",parameters, "trainlosses", trainlosses, "ode_data", ode_data)

D = load("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_NeuralUDE.jld")
parameters = D["parameters"]
trainlosses = D["trainlosses"]

### Universal ODE Part 2: SInDy to Equations
using DataDrivenDiffEq
using ModelingToolkit
# Create a Basis
@variables u[1:2]
# Lots of polynomials
polys = Operation[]
for i ∈ 0:2, j ∈ 0:2
    push!(polys, u[1]^i * u[2]^j )
end

h = [unique(polys)...]
basis = Basis(h, u)

#opt = SR3() #STRRidge(1e-3)
#opt = SR3(0.5) works good, but maybe is too sparse

#opt = SR3(0.5)

opt = STRRidge(3)
# Create the thresholds which should be used in the search process
#thresholds = exp10.(-6:0.1:0)


for k in 1:300
    x_sol, y_sol, UDE_SGLD = UDE_term(UDE_SGLD, rand(1:300))
	Ψ = SInDy(ode_data[1:2, 1:end], UDE_SGLD[1:end], basis, opt = opt, maxiter = 10000, normalize = true, denoise = true) # Suceed
    plot!(x_sol, y_sol,UDE_SGLD[1:end], alpha=0.04, color = :red, label = "")
end

aicc_store = zeros(1, 100)

for i in 1:100
x_sol, y_sol, UDE_SGLD = UDE_term(UDE_SGLD, i)
Ψ = SInDy(vcat(x_sol', y_sol'), UDE_SGLD[1:end], basis, opt = opt, maxiter = 1000, normalize = true, denoise = true) # Suceed
aicc_store[i] = Ψ.aicc[1]
end

###5 -- -1.02
####3 - 4.095
####2 - 7.236
####1 - 21.63
####0.1 - 35
####0.01 - 40.4


x_sol, y_sol, UDE_SGLD = UDE_term(UDE_SGLD, findmin(aicc_store)[2][2])
Ψ = SInDy(vcat(x_sol', y_sol'), UDE_SGLD[1:end], basis, opt, maxiter = 10000, normalize = true, denoise = true) # Suceed

print_equations(Ψ,  show_parameter = true)

########STRRIDGE PLOTS######################
using LaTeXStrings
x=[0.01; 0.1; 1; 2; 3; 5]
y=[9; 9; 5; 2; 1; 1]
 z=[76; 76; 77; 63; 70; 100]

 plot([0.01, 1.5], [80, 80], xscale = :log, fill=(0, :lightpink), markeralpha=0, label = "")
 plot!([3.5, 5.5], [80, 80],xscale = :log,fill=(0,:lightpink), markeralpha=0, label = "", framestyle = :box, ylims = (0, 10))
 plot!([1.5, 3.5], [80, 80],xscale = :log, fill=(0,:aliceblue), markeralpha=0, label = "", framestyle = :box, ylims = (0, 10))

 scatter(x,y,marker_z=z,  label = "", xlabel = L"\lambda", xscale = :log, ylabel  = "Number of Active terms",framestyle = :box, color = :algae, ylims = (0, 10), markersize = 8, colorbar_title = "100* Error")

 Plots.savefig("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_LV_Plots5_SINDY.pdf")



#= GET ERROR SIMULATION###########
u1 = x_sol
u2 = y_sol

Term1 = 0.0169 * u1.^2 + 0.839*u2 + 0.075*(u1.^2) .* u2 + 0.2555* u1 .* u2

scatter(Term1)

scatter!(UDE_SGLD)

norm(Term1 .- UDE_SGLD)
