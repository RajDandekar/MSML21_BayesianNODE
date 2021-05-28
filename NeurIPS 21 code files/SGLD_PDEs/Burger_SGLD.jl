# 1D diffusion problem

# Packages and inclusions
cd("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE")
Pkg.activate(".")
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using JLD
using Random
using ModelingToolkit,DiffEqOperators,DiffEqBase,LinearAlgebra

# Parameters, variables, and derivatives
@parameters t x
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# 1D PDE and boundary conditions
eq  = Dt(u(t,x)) ~ (0.01/π)*Dxx(u(t,x)) - u(t,x) * Dx(u(t,x))

bcs = [u(0,x) ~ -sin(pi*x),
       u(t,-5) ~ 0.0,
       u(t,5) ~ 0.0]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(-5.0,5.0)]

# PDE system
pdesys = PDESystem(eq,bcs,domains,[t,x],[u])

# Method of lines discretization
dx = 0.2
order = 3
discretization = MOLFiniteDifference(dx,order)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys,discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob,Tsit5(),saveat=0.02)

sol = hcat(Array(sol.u)...)
z1 = zeros(1,length(sol[1,:]))
sol = vcat(z1, sol, z1)

training_data = sol
xgrid = -5:0.2:5
tgrid = 0:0.02:1

using PyPlot
fig = figure()

cmap = get_cmap("RdBu")
pcolormesh(xgrid,tgrid,sol', cmap = cmap, rasterized=true)
xticks([], [])
ylabel(L"$t$"); title("Data")
xlabel(L"$x$")

#yticks([0, 1, 2, 3, 4, 5])
gcf()

################THE TRAINING BEGINS####################
tspan = (0.0, 1.0)

dudt2 = FastChain(FastDense(51, 10, relu),
                  FastDense(10, 51))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tgrid)
p= prob_neuralode.p
u0 = - sin.(π*xgrid)

pred = Array(prob_neuralode(u0, p))

loss = sum(abs2, training_data .- pred)

function train_neuralsir(steps, a, b, γ)
    θ = p

    predict(p) = Array(prob_neuralode(u0, p))

	loss(p) =     sum(abs2, training_data -predict(p))


	#loss(p) = sum(abs2, ode_data .- pred)

	trainlosses = [loss(p); zeros(steps)]

	weights = [p'; zeros(steps, length(p))]

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

	############### NO PRECONDITIONING#####################
	#=
	 if t == 1
	   precond[:] = ∇L.*∇L
     else
	   precond *= beta
	   precond += (1-beta)*(∇L .*∇L)
     end
	 #m = λ .+ sqrt.(precond/((1-(beta)^t)))
	 m = λ .+ sqrt.(precond)
=#
	 ###############DESCENT###############################
	 for i in 1:length(∇L)
		 noise = ϵ*randn()
		 ###Implement Sam Brand's suggestion########
		 #noise = sqrt(ϵ/m[i])*randn()
		 θ[i] = θ[i] - (0.5*ϵ*∇L[i] +  noise)
		 #θ[i] = θ[i] - (0.5*ϵ*∇L[i]/m[i] +  noise)
	 end

	#################BOOKKEEPING############################
	weights[t+1, :] = θ
	trainlosses[t+1] = loss(θ)
	println(loss(θ))
	end
    print("Final loss is $(trainlosses[end])")

    trainlosses, weights
end

#results =  train_neuralsir(6000, 0.0000006)

#results =  train_neuralsir(6000, 0.0000006, 10, 1e-6, 0.9)

results =  train_neuralsir(40000, 0.001, 0.15, 0.05)

trainlosses, parameters_burger = results;

using JLD2, LaTeXStrings

JLD2.@save "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Burger_Data_SGLD.jld2" results training_data trainlosses  parameters_burger

JLD2.@load "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Burger_Data_SGLD.jld2" results training_data trainlosses  parameters_burger

function predict_rd(p)
      Array(prob_neuralode(u0, p))
end


######################COLORMESH##################################
using PyPlot

fig = figure()

fig.subplots_adjust(right=0.45)
fig.subplots_adjust(hspace=0.45)

subplot(311)
cmap = get_cmap("RdBu")

img = pcolormesh(xgrid,tgrid,training_data',rasterized=true)
xticks([], [])
yticks([], [])

ylabel(L"$t$"); title(string("Burgers equation: ",  L"$u_{t} = (0.01/\pi) u_{xx} - u u_{x}$",  "\n", "Data"))
cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$u$")
clim([0, 1]);
colb.set_ticks([0, 1])

#yticks([0, 1, 2, 3, 4, 5])


ax = subplot(312)
cur_pred = predict_rd(parameters_burger[end-rand(1:600), :])

for k in 1:500
cur_pred += predict_rd(parameters_burger[end-rand(1:600), :])
end

cur_pred = (1/501)*cur_pred

img = pcolormesh(xgrid,tgrid,cur_pred',rasterized=true)
global img
ylabel(L"$t$"); title("Mean Posterior Prediction")
#yticks([0, 0.5, 1, 1.5])
xticks([], [])

cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$u$")
clim([0, 1]);
colb.set_ticks([0, 1])

ax = subplot(313)
error = abs.(training_data - cur_pred)

img = pcolormesh(xgrid,tgrid,error',rasterized=true)
global img
xlabel(L"$x$"); ylabel(L"$t$"); title("Mean Error")
yticks([], [])

#yticks([0, 0.5, 1, 1.5])
cax = fig.add_axes([0.52,0.1,.02,.19])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$Error$")
clim([0, 0.05]);
colb.set_ticks([0, 0.05])


gcf()

fig.savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Burger_Colormesh_noresize_SGLD.pdf")
