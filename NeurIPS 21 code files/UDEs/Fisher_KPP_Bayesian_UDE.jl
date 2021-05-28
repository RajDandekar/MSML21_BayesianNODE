#############SHOWING THAT SEIR ALSO WORKS WITH SGLD APPROACH!
#########FOR DIFF LAMBDA AND FOR ALL ITERATIONS, SAME TERMS RECOVERED.
#####NEED TO CAPTURE AICC FOR DIFF LAMBDA AND 100 ITERATIONS#######

cd("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE")
Pkg.activate(".")

using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using JLD
using Random

#parameter
D = 0.01; #diffusion
r = 1.0; #reaction rate

#domain
X = 1.0; T = 5;
dx = 0.04; dt = T/10;
x = collect(0:dx:X);
t = collect(0:dt:T);
Nx = Int64(X/dx+1);
Nt = Int64(T/dt+1);

#initial conditions
Amp = 1.0;
Delta = 0.2
#IC-1
rho0 = Amp*(tanh.((x .- (0.5 - Delta/2))/(Delta/10)) - tanh.((x .- (0.5 + Delta/2))/(Delta/10)))/2
#IC-2
#rho0 = Amp*(1 .- tanh.((x .- 0.2)/(Delta/6)))/2

########################
# Generate training data
########################
reaction(u) = r * u .* (1 .- u)
lap = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2
#Periodic BC
lap[1,end] = 1.0/dx^2
lap[end,1] = 1.0/dx^2
#Neumann BC
#lap[1,2] = 2.0/dx^2
#lap[end,end-1] = 2.0/dx^2

function rc_ode(rho, p, t)
    #finite difference
    D * lap * rho + reaction.(rho)
end

prob = ODEProblem(rc_ode, rho0, (0.0, T), saveat=dt)
sol = solve(prob, Tsit5());

ode_data = hcat([sol[:,i]  for i in 1:size(sol,2)]...)



######DEFINE THE UDE##########
n_weights = 10

#for the reaction term
rx_nn = Chain(Dense(1, n_weights, relu),
                Dense(n_weights, 1),
                x -> x[1])

#conv with bias with initial values as 1/dx^2
w_err = 0.0
init_w = reshape([1.1 -2.5 1.0], (3, 1, 1, 1))
diff_cnn_ = Conv(init_w, [0.], pad=(0,0,0,0))

#initialize D0 close to D/dx^2
D0 = [6.5]

p1,re1 = Flux.destructure(rx_nn)
p2,re2 = Flux.destructure(diff_cnn_)
p = [p1;p2;D0]
full_restructure(p) = re1(p[1:length(p1)]), re2(p[(length(p1)+1):end-1]), p[end]


function nn_ode(u,p,t)
    rx_nn = re1(p[1:length(p1)])

    u_cnn_1   = [p[end-4] * u[end] + p[end-3] * u[1] + p[end-2] * u[2]]
    u_cnn     = [p[end-4] * u[i-1] + p[end-3] * u[i] + p[end-2] * u[i+1] for i in 2:Nx-1]
    u_cnn_end = [p[end-4] * u[end-1] + p[end-3] * u[end] + p[end-2] * u[1]]

    # Equivalent using Flux, but slower!
    #CNN term with periodic BC
    #diff_cnn_ = Conv(reshape(p[(end-4):(end-2)],(3,1,1,1)), [0.0], pad=(0,0,0,0))
    #u_cnn = reshape(diff_cnn_(reshape(u, (Nx, 1, 1, 1))), (Nx-2,))
    #u_cnn_1 = reshape(diff_cnn_(reshape(vcat(u[end:end], u[1:1], u[2:2]), (3, 1, 1, 1))), (1,))
    #u_cnn_end = reshape(diff_cnn_(reshape(vcat(u[end-1:end-1], u[end:end], u[1:1]), (3, 1, 1, 1))), (1,))

    [rx_nn([u[i]])[1] for i in 1:Nx] + p[end] * vcat(u_cnn_1, u_cnn, u_cnn_end)
end

########################
# Soving the neural PDE and setting up loss function
########################
prob_nn = ODEProblem(nn_ode, rho0, (0.0, T), p)

sol_nn = Array(concrete_solve(prob_nn,Tsit5(), rho0, p, saveat = dt))


function train_neuralsir(steps, a, b, γ)
    θ = p

    predict(p) =  Array(concrete_solve(prob_nn,Tsit5(),rho0,p,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))

	loss(p) = sum(abs2, ode_data .- predict(p)) + 10^2 * abs(sum(p[end-4 : end-2]))

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

results =  train_neuralsir(20000, 0.001, 0.15, 0.05)


trainlosses, parameters = results;

save("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/FisherKPPUDE_loss_regularized.jld", "results", results, "ode_data", ode_data, "trainlosses", trainlosses, "parameters", parameters)

Dict_NN = load("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/FisherKPPUDE_loss_regularized.jld")

parameters_fkpp = Dict_NN["parameters"]
trainlosses = Dict_NN["trainlosses"]

#save("C:/Users/16174/Desktop/Julia Lab/Bayesian Neural UDE/SGLD_ADAM_SEIR_morelayers_NeuralUDE.jld", "parameters",parameters, "trainlosses", trainlosses, "ode_data", ode_data)
function predict_rd(p)
	#Array(concrete_solve(prob_nn,Tsit5(),rho0,p,saveat=dt,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
Array(concrete_solve(prob_nn,Tsit5(),rho0,p,saveat=dt))
end


######################COLORMESH##################################

using PyPlot
fig = figure()

fig.subplots_adjust(right=0.45)
fig.subplots_adjust(hspace=0.45)

subplot(311)
img = pcolormesh(x,t,ode_data', rasterized=true)
xticks([], [])
yticks([], [])

ylabel(L"$t$"); title(string("Fisher KPP equation: ",  L"$\rho_{t} = \rho (1 - \rho) + D \rho_{xx}$",  "\n", "Data"))
cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\bar{T}$")
clim([0, 1]);
colb.set_ticks([0, 1])

#yticks([0, 1, 2, 3, 4, 5])


ax = subplot(312)
cur_pred = predict_rd(parameters_fkpp[end-rand(1:600), :])

for k in 1:500
cur_pred += predict_rd(parameters_fkpp[end-rand(1:600), :])
end

cur_pred = (1/501)*cur_pred

img = pcolormesh(x,t,cur_pred',rasterized=true)
global img
ylabel(L"$t$"); title("Mean Posterior Prediction")
#yticks([0, 0.5, 1, 1.5])
xticks([], [])

cax = fig.add_axes([0.52,0.45,.02,.29])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$\rho$")
clim([0, 1]);
colb.set_ticks([0, 1])

ax = subplot(313)
error = abs.(ode_data - cur_pred)

img = pcolormesh(x,t,error',rasterized=true)
global img
xlabel(L"$x$"); ylabel(L"$t$"); title("Mean Error")
yticks([], [])

#yticks([0, 0.5, 1, 1.5])
cax = fig.add_axes([0.52,0.1,.02,.19])
colb = fig.colorbar(img, cax=cax)
colb.ax.set_title(L"$Error$")
clim([0, 0.1]);
colb.set_ticks([0, 0.1])


gcf()


PyPlot.savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/FKPP_Colormesh_normal.pdf")


##########################REACTION TERM#####################################
full_restructure(p) = re1(p[1:31]), re2(p[32:end-1]), p[end]


using Plots

u = collect(0:0.01:1)

Plots.plot()

pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
rx_array = rx_nn.([[elem] for elem in u])

for k in 1:1:1000
pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
rx_array = rx_array + rx_nn.([[elem] for elem in u])
Plots.plot!(u, rx_nn.([[elem] for elem in u]), alpha=0.04, color = :red, label = "", ylims = (-0.1, 0.6))[1];
end

rx_array_mean = (1/1001)*(rx_array)
Plots.scatter!(u, reaction.(u), color = :blue, label = "Reaction term - True", xlabel = "u", title = "Fisher-KPP Reaction term Bayesian recovery")

p1_new = Plots.plot!(u,rx_array_mean, color = :black, w = 2, label = "Mean prediction")

savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Reaction_Term.pdf")

#############################WEIGHT TERMS##########################################
metric1 = Float64[]
metric2 = Float64[]

for k in 1:1:1000
pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
weight = diff_cnn_.weight[:]
w1 = weight[1]
w2 = weight[2]
w3 = weight[3]
append!(metric1, w1/w3)
append!(metric2, w1 + w2 + w3)
end

#=
p2 = plot()
histogram!(metric1, label="w1/w3", color=:blue, alpha=0.5, title = "Fisher-KPP stencil Bayesian recovery")
savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Stencil1.pdf")
=#

Plots.plot()

Plots.scatter!(metric1, label= L"w_{1}/w_{3}", ylims = (0.8, 1.25), color=:blue, alpha=0.5, title = "Fisher-KPP stencil Bayesian recovery")
f(x) = 1
p2_new = plot!(f, xlims = (0,1010), color = :blue, linestyle = :dash, linewidth = 3, label = L"w_{1}/w_{3} = 1")

savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Stencil1_Scatter.pdf")

#=
p3 = plot()
histogram!(metric2, label="w1 + w2 + w3", color=:red, alpha=0.5, title = "Fisher-KPP stencil Bayesian recovery")
savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Stencil2.pdf")
=#

Plots.plot()


Plots.scatter!(metric2, label= L"w_{1} + w_{2} + w_{3}", xlims = (0, 1000), ylims = (-0.05, 0.05), color=:red, alpha=0.5, title = "Fisher-KPP stencil Bayesian recovery")
f(x) = 0
p3_new = plot!(f, xlims = (0,1010), color = :black, linestyle = :dash, linewidth = 3, label = L"w_{1} + w_{2} + w_{3} = 0")

savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/Stencil2_Scatter.pdf")

gr()

l = @layout [a; b; c]

Plots.plot(p1_new, p2_new, p3_new, layout = l)


#############################DIFFUSION TERM##########################################
diffusion_term = Float64[]
for k in 1:1:1000
pstar = parameters[end-rand(1:1000), :]
rx_nn, diff_cnn_, D0 = full_restructure(pstar)
append!(diffusion_term, D0*dx*dx)
end

p4 = plot()
histogram!(diffusion_term, label="Diffusion term (true value = 0.01)", color=:green, alpha=0.5, title = "Fisher-KPP diffusion term Bayesian recovery", ylims = (0, 250))


savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/DiffusionTerm.pdf")



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
