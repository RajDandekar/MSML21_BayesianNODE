#############SHOWING THAT SEIR ALSO WORKS WITH SGLD APPROACH!
#########FOR DIFF LAMBDA AND FOR ALL ITERATIONS, SAME TERMS RECOVERED.
#####NEED TO CAPTURE AICC FOR DIFF LAMBDA AND 100 ITERATIONS#######

cd("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE")
Pkg.activate(".")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using JLD
using Random
const EIGEN_EST = Ref(0.0f0)
const USE_GPU = Ref(false)

USE_GPU[] = false # the network is small enough such that CPU works just great

USE_GPU[] && (using CUDA)

_gpu(arg) = USE_GPU[] ? gpu(arg) : cpu(arg)
_cu(arg) = USE_GPU[] ? cu(arg) : identity(arg)

function getops_new(grid, T=Float32)
    N, dz = length(grid), step(grid)
    d = ones(N-2)
    dl = ones(N-3) # super/lower diagonal
    zv = zeros(N-2) # zero diagonal used to extend D* for boundary conditions

    # D1 first order discretization of ∂_z
    D1= diagm(-1 => -dl, 0 => d)
    D1_B = hcat(zv, D1, zv)
    D1_B[1,1] = -1
    D1_B = _cu((1/dz)*D1_B)

    # D2 discretization of ∂_zz
    D2 = diagm(-1 => dl, 0 => -2*d, 1 => dl)
    κ = 0.05
    D2_B = hcat(zv, D2, zv) #add space for the boundary conditions space for "ghost nodes"
    #we only solve for the interior space steps
    D2_B[1,1] = D2_B[end, end] = 1

    D2_B = _cu((κ/(dz^2)).*D2_B) #add the constant κ as the equation requires and finish the discretization

    # Boundary conditions matrix QQ
    Q = Matrix{Int}(I, N-2, N-2)
    QQ = _cu(vcat(zeros(1,N-2), Q, zeros(1,N-2)))

    D1 = D1_B * QQ
    D2 = D2_B * QQ
    EIGEN_EST[] = maximum(abs, eigvals(Matrix(D2)))
    return (D1=T.(D1), D2=T.(D2))
end

function getu0(grid, T=Float32)
    z = grid[2:N-1]
    f0 = z -> T(exp(-200*(z-0.75)^2))
    u0 = f0.(z) |> _gpu
end

function ode_i(u, p, t)
    Φ = u -> cos.(sin.(u.^3) .+ sin.(cos.(u.^2)))
    return p.D1*Φ(u) + p.D2*u
end

function ground_truth(grid, tspan)
    prob = ODEProblem(ode_i, u0, (tspan[1], tspan[2]+0.1), ops)
    sol = solve(prob, ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]), abstol = 1e-9, reltol = 1e-9)
    return sol
end

N = 32
grid = range(0, 1, length = N)
tspan = (0.0f0, 1.5f0)
u0 = getu0(grid)
ops = getops_new(grid)
soldata = ground_truth(grid, tspan)




ann = Chain(Dense(30,8,tanh), Dense(8,30,tanh)) |> _gpu
p, re = Flux.destructure(ann)
lyrs = Flux.params(p)
function dudt_(u,p,t)
    Φ = re(p)
    return ops.D1*Φ(u) + ops.D2*u
end

saveat = range(tspan..., length = 30) #time range
prob = ODEProblem{false}(dudt_,u0,tspan,p)
training_data = _cu(soldata(saveat))

#concrete_solve(prob, ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]), u0, pp)



function train_neuralsir(steps, a, b, γ)
    θ = p

    predict(p) =  Array(concrete_solve(prob,
	                    ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]),u0, p, saveat = saveat))

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
		 ###Implement Sam Brand's suggestion########
		 #noise = sqrt(ϵ/m[i])*randn()
		 #θ[i] = θ[i] - (0.5*ϵ*∇L[i] +  noise)
		 θ[i] = θ[i] - (0.5*ϵ*∇L[i]/m[i] +  noise)
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

results =  train_neuralsir(50000, 0.001, 0.15, 0.05)


trainlosses, parameters = results;

save("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Climate_UDE.jld", "results", results, "training_data", training_data, "trainlosses", trainlosses, "parameters", parameters)

Dict_NN = load("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Climate_UDE.jld")

parameters_climate = Dict_NN["parameters"]
trainlosses = Dict_NN["trainlosses"]

#save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_SEIR_morelayers_NeuralUDE.jld", "parameters",parameters, "trainlosses", trainlosses, "ode_data", ode_data)
function predict_rd(p)
	#Array(concrete_solve(prob_nn,Tsit5(),rho0,p,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
	Array(concrete_solve(prob,
						ROCK4(eigen_est = (integ)->integ.eigen_est = EIGEN_EST[]),u0, p, saveat = saveat))

end

z = grid[2:N-1]


######################COLORMESH#################################
using PyPlot

fig = figure()

fig.subplots_adjust(right=0.45)
fig.subplots_adjust(hspace=0.45)

subplot(311)
img = pcolormesh(z,saveat,training_data',rasterized=true)
xticks([], [])
yticks([], [])

ylabel(L"$t$"); title(string("Temperature convection equation: ",  L"$\bar{T}_{t} = - (w \bar{T})_{z} - \kappa \bar{T}_{zz}$",  "\n", "Data"))
cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\bar{T}$")
clim([0, 1]);
colb.set_ticks([0, 1])

#yticks([0, 1, 2, 3, 4, 5])


ax = subplot(312)
cur_pred = predict_rd(parameters_climate[end-rand(1:600), :])

for k in 1:500
cur_pred += predict_rd(parameters_climate[end-rand(1:600), :])
end

cur_pred = (1/501)*cur_pred

img = pcolormesh(z,saveat,cur_pred',rasterized=true)
global img
ylabel(L"$t$"); title("Mean Posterior Prediction")
#yticks([0, 0.5, 1, 1.5])
xticks([], [])

cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\bar{T}$")
clim([0, 1]);
colb.set_ticks([0, 1])

ax = subplot(313)
error = abs.(training_data - cur_pred)

img = pcolormesh(z, saveat,error',rasterized=true)
global img
xlabel(L"$z$"); ylabel(L"$t$"); title("Mean Error")
yticks([], [])

#yticks([0, 0.5, 1, 1.5])
cax = fig.add_axes([0.52,0.1,.02,.19])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$Error$")
clim([0, 0.05]);
colb.set_ticks([0, 0.05])


gcf()



PyPlot.savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Climate_Colormesh_normal.pdf")


##########################SPATIAL PLOT#####################################
n = size(training_data, 1)
cur_pred = predict_rd(parameters[end-rand(1:1200), :])
pl = Plots.plot(1:n,cur_pred[:,10], color =:purple, alpha = 0.6, label="prediction",title="Spatial Plot at t=$(saveat[10])")

for k in 1:1000
	cur_pred = predict_rd(parameters[end-rand(1:1200), :])
	plot!(1:n,cur_pred[:,10], color =:purple, alpha = 0.008, label="")
end

Plots.scatter!(1:n,training_data[:,10],color =:black, label="data", legend =:topright)


display(pl)

Plots.savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Climate_TemporalSlice.pdf")

##########################TEMPORAL PLOT#####################################
cur_pred = predict_rd(parameters[end-rand(1:1200), :])
pl2 = Plots.plot(saveat,cur_pred[N÷2,:],color =:purple, alpha = 0.6, label="prediction")

for k in 1:1000
cur_pred = predict_rd(parameters[end-rand(1:1200), :])
plot!(saveat,cur_pred[N÷2,:],color =:purple, alpha = 0.008, label="")

end

Plots.scatter!(saveat,training_data[N÷2,:],label="data", legend =:bottomright, color =:black, title="Time Series Plot: Middle X")

display(pl2)

Plots.savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Climate_SpatialSlice.pdf")


###############SINDY ON REACTION TERM###############################
### Universal ODE Part 2: SInDy to Equations
using DataDrivenDiffEq
using ModelingToolkit
# Create a Basis
@variables x
u = [x]

# Lots of polynomials
polys = Any[]
for i ∈ 0:3
    push!(polys, u[1]^i)
end


h = [unique(polys)...]
basis = Basis(polys, u)

###################After this point, you have to use the following steps#########

# Step1: Find the optimal λ:

###Generally the optimal λ is the one with lowest positive aicc score.
#####Note that the j loop times out after 100 iteration, I have not fully optimized it yet. I am currently running it till 20 in this code
####Note that ultimately we want the least amount of active terms.

λ = [0.005, 0.05, 0.1, 0.5, 2, 4, 4.15]
AICC_Score = zeros(1,7)
Error_Score = zeros(1, 7)
Count_1 = zeros(1, 7)
Count_2 = zeros(1,7)
Count_3 = zeros(1, 7)

for i in 1:1:length(λ)

opt = STRRidge(λ[i])
###The below works
#=
opt = STRRidge(0.5)
=#

#opt = STRRidge(0.05)
aicc_store = 0
error_store = 0
count1_store = 0
count2_store = 0
count3_store = 0

u = collect(0:0.01:1)


#=
count1 = 0
count2 = 0
=#
for j in 1:100
pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
rx_sol = rx_nn.([[elem] for elem in u])
Ψ = SInDy(u, rx_sol, basis, opt = opt, maxiter = 10000, normalize = true) # Suceed
aicc_store += Ψ.aicc[1]
error_store += get_error(Ψ)[1]

if Ψ.sparsity[1] == 1
	count1_store+=1
end

if Ψ.sparsity[1] == 2
	count2_store+=1
end

if Ψ.sparsity[1] == 3
	count3_store+=1
end

#=
if Ψ.sparsity[1] == 2.0
	count1+=1
end

if Ψ.sparsity[1] == 3
	count2+=1
end
=#
end

AICC_Score[i] = aicc_store/100
Error_Score[i] = error_store/100
Count_1[i] = count1_store
Count_2[i] = count2_store
Count_3[i] = count3_store
end


Plots.scatter(λ,AICC_Score', label = "", xlabel =  L"\lambda", xscale = :log,ylabel  = "AICC Score",framestyle = :box, color = :blue, ylims = (6, 20), markersize = 8)

plot!([0.5-0.001,0.5+0.001],[6, 18],lw=3,color=:green, label=L"\lambda_{cr}",linestyle = :dash, labelsize = 10)

savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/AICC_Score_Plot.pdf")


###0.005 - 22.1172
###0.05 - 22.118
###0.1 - 22.1184
###0.5 - 0 terms recovered


####Step 2:  Note down terms for all runs. In this case, I get the same terms each time######
for i in 1:100
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
rx_sol = rx_nn.([[elem] for elem in u])
Ψ = SInDy(u, rx_sol, basis, opt = opt, maxiter = 10000, normalize = true) # Suceed

print_equations(Ψ,  show_parameter = true)
end


###############PARETO FRONTS: SINDY###########################


opt = STRRidge(0.5)
λ = Float32.(0.05: 0.05: 5)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
g(x) = x[1] < 1 ? Inf : norm(x, 2)

count1 = 0
count2 = 0

for j in 1:100
pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
rx_sol = rx_nn.([[elem] for elem in u])
Ψ = SINDy(u, rx_sol, basis, λ, opt, g = g, maxiter = 10000) # Succeed

if Ψ.sparsity[1] == 2.0
	count1+=1
end

if Ψ.sparsity[1] == 3
	count2+=1
end
end
