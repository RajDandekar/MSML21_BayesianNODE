cd(dirname(@__FILE__))
using Pkg; Pkg.activate(".")
using Random, Distributions, LinearAlgebra, Statistics
using Plots, Plots.PlotMeasures, StatsPlots, LaTeXStrings
using Zygote, Flux, Turing, MCMCChains, AdvancedMH, Optim, AdvancedHMC
using DistributionsAD, AdvancedVI
using DiffEqFlux, DifferentialEquations
include("utils.jl")

#============================================#
# Defining the model and generating data
#============================================#
function lotka_volterra!(du, u, p, t)
  α, δ, β, γ = p
  x, y = u

  du[1] = dx = α*x - β*x*y
  du[2] = dy = - δ*y + γ*x*y
end


ptrue = [1.5, 1.0, 3.0, 1.0] # α, β, δ, γ

u0 = [1.0, 1.0]
tspan = (0.0, 2.0)
tsteps = 0.0:0.1:2.0
Nt = size(tsteps)[1]

prob_ode = ODEProblem(lotka_volterra!, u0, tspan, ptrue)
data = Array(solve(prob_ode, Tsit5(), u0=u0, p=ptrue, saveat=tsteps))

data_x, data_y = get_states_from_array(data)
data_vectorized = vectorize_observation( data )

function lotka_volterra_ude!(du, u, p, t)
  α, δ, γ = abs.(p[NN_Np+1:end])
  x, y = u
  NN1 = abs(ann([x;y], p[1:NN_Np])[1])

  # TODO: Any way to "normalize" the contribution of NN?
  # e.g. if we know there's a reasonable range of values this
  # term should take can we scale it appropriately?
  du[1] = dx = α*x - NN1
  du[2] = dy = - δ*y + γ*x*y
end

# The parameters go like: the first 2x5 params are the weights,
# the next 5 are the biases; the next 5x1 params are the weights,
# the next 1 param is the bias.
Random.seed!(1)
ann = FastChain(FastDense(2,3,relu),  FastDense(3,1))
NN_p = initial_params(ann)
NN_Np = size(NN_p)[1]

model_p = [2.0, 2.0, 2.0] # α, δ, γ prior means
pinit = [NN_p; model_p]

# Setup the ODE problem, then solve
prob_ude = ODEProblem(lotka_volterra_ude!, u0, tspan, p)

function physics_model( p )
    sol = Array(solve(prob_ude, Tsit5(), u0=u0, p=p, saveat=tsteps))
    if !(size(sol)[2]==length(tsteps))
        display(p)
        error("ODE solver did not complete for the set of params displayed.")
    end
    sol
end

function prior()

    # Using the priors that worked for just inferring model params
    # in bayesian_inference_LV.jl.
    α_dist = Normal(2,0.5)
    δ_dist = Normal(2,0.5)
    γ_dist = Normal(2,0.5)

    # NOTE: Currently Distributions.jl doesn't support product
    # distributions of multivariate distributions with univariate
    # distributions. Instead we'll define an individual normal
    # distribution for each NN param
    #NNp_dist = MvNormal(zeros(NN_Np),σ₀²*I)
    #Product([NNp_dist, α_dist, δ_dist, γ_dist])

    σ₀² = 0.01
    σ₀ = σ₀²^0.5
    d = Normal(0,σ₀)
    dist_vec = vcat([ d for i in 1:NN_Np], [ α_dist, δ_dist, γ_dist])
    Product(dist_vec)
end

# This prior instantiation is used in some of the training
# methods and log posterior
prior_inst = prior()

function generate_prior_samples(n_samples)
    samples = transpose(rand( prior_inst, n_samples ))
    chain = get_MCMCChain_from_raw_samples( samples )
end

function likelihood(p)
    sol_vectorized = vectorize_observation(physics_model(p))
    σ = 0.01
    MvNormal(sol_vectorized, σ^2*I) # This will produce an isotropic normal with variance σ²
end

function logpost(p,y)
    # y is the observed value used in inference
    logpdf(likelihood(p), y) + logpdf(prior_inst, p)
end

function logpostp( p )
    logpost(p,data_vectorized)
end

#=
Info on julia macros, denoted by the @ symbol, here
https://docs.julialang.org/en/v1/manual/metaprogramming/#man-macros
=#

@model target(y) = begin

    # If y passed as missing it defaults
    # to sampling y from the distribution
    # see https://turing.ml/dev/docs/using-turing/guide
    if y === missing
        # If y passed as missing, instantiate
        y = Vector{typeof(data_vectorized)}(undef, size(data_vectorized))
    end

    p ~ prior()
    y ~ likelihood(p)
end

#==============================================================================#
# Prior predictive check
# https://storopoli.io/Bayesian-Julia/pages/4_Turing/#prior_and_posterior_predictive_checks
# https://turing.ml/dev/docs/using-turing/guide#sampling-from-an-unconditional-distribution-the-prior
# The target distribution can be turned into a "predictive model" by setting
# the observations to "missing;" this tells Turing to sample from the likelihood
# rather than evaluate it at the passed value y.
#==============================================================================#

# Documentation on manipulating turing Chains objects:
# https://turinglang.github.io/MCMCChains.jl/dev/chains/

function plot_data!()
    plot!(tsteps, data_x, seriestype = :scatter, color = 1, label="x pop data")
    plot!(tsteps, data_y, seriestype = :scatter, color = 2, label="y pop data")
    plot!(ylim=(0,1.1*max(maximum(data_x),maximum(data_y))))
end
function plot_data()
    plot()
    plot_data!()
end

function plot_predictive_samples( samples )
    plot() # Clear out and start new plot
    raw_samples = get_raw_samples_from_MCMCChain(samples)
    for i in 1:size(raw_samples)[1]
        x_sample, y_sample = get_states_from_vector( raw_samples[i,:], Nt )

        if i==1
            plot!(tsteps, x_sample, color=1, alpha=0.2, label="x pop samples")
            plot!(tsteps, y_sample, color=2, alpha=0.2, label="y pop samples")
        else
            plot!(tsteps, x_sample, color=1, alpha=0.2, label="")
            plot!(tsteps, y_sample, color=2, alpha=0.2, label="")
        end
    end
    plot!()
end
function plot_predictive_quantiles( samples )
    x_mean, y_mean = get_states_from_vector( mean(samples).nt.mean, Nt)
    x_5pct, y_5pct = get_states_from_vector( quantile(samples, q=[0.05]).nt[2], Nt )
    x_95pct, y_95pct = get_states_from_vector( quantile(samples, q=[0.95]).nt[2], Nt )
    plot()
    plot!(tsteps, x_mean, ribbon=[x_mean-x_5pct, x_95pct-x_mean], fillalpha=0.3, color=1, label="x pop mean and quantiles")
    plot!(tsteps, y_mean, ribbon=[y_mean-y_5pct, y_95pct-y_mean], fillalpha=0.3, color=2, label="y pop mean and quantiles")
end

predictive_target = target(missing)

N_prior_samples = 100
prior_samples = generate_prior_samples( N_prior_samples )
# Stripping out raw data and then converting back to MCMCCHain to avoid an
# annoying warning that MCMCChains produces given the way predict() formats
# the MCMCChain it returns.
priorp_samples = prior_predictive_samples =
    get_MCMCChain_from_raw_samples(get_raw_samples_from_MCMCChain(
    predict( predictive_target, prior_samples )))

plot_predictive_samples(priorp_samples)
plot_predictive_quantiles(priorp_samples)
plot_data!()

#===========================================================#
# Setting initial guess, importing training methods
#===========================================================#
include("training_methods.jl")
post_inst = target(data_vectorized)
map_estimate = optimize(post_inst, MAP())
mle_estimate = optimize(post_inst, MLE())

function plot_model_vs_data( p )
    plot_data()
    model_eval = Array(solve(prob_ode, Tsit5(), u0=u0, p=p, saveat=tsteps))
    plot!(tsteps, model_eval[1,:], color=1, label="Modeled x pop")
    plot!(tsteps, model_eval[2,:], color=2, label="Modeled y pop")
end

plot_model_vs_data( map_estimate.values.array )
# Stripping out the raw data from the ModeResult data structure
# returned by optimize() above
p0 = copy(map_estimate.values.array)

#==============================================================================#
# INFERENCE STARTS HERE
#==============================================================================#
# NOTE: we have so many params in this case that it might overload this
# built-in plotting method. May want to do something that's just a summary
# stat, like just the hists, and one per param?
function plot_chain_vs_true(chain, ptrue)
    plt = plot(chain)
    for i=1:length(ptrue)
        hline!([ptrue[i]], subplot=2*i-1)
        vline!([ptrue[i]], subplot=2*i)
    end
    plot!()
end

#==============================================================================#
#  Variational inference method
#==============================================================================#
samples, posterior_approx = run_ADVI_training(p0, 10000;
    samples_per_step=10, max_iters=1000)

# NOTE: probably don't want to call this for the UDE case. Refactor diagnostic
# plotting for this case with so many params.
plot_chain_vs_true(samples, ptrue)

#==============================================================================#
#  MCMC methods
#==============================================================================#

chain, stats = run_NUTS_training( p₀, 2000 )
subset_chain = MCMCChains.subset(chain, 250:size(chain)[1])
plot_chain_vs_true(subset_chain, ptrue)

chain, logpost_values = run_SGLD_training(p₀, 10000;
    a=1e-2, b=0.85, γ=.6)
#chain, logpost_values = run_SGLD_training(p0, 20000;
#    a=1e-4, b=0.85, γ=0.5)
#subset_chain = MCMCChains.subset(chain, 2500:size(chain)[1])
#plot_chain_vs_true(chain, ptrue)

chain, logpost_values = run_PSGLD_training(p₀, 10000;
    a=1e-2, b=0.85, γ=1e-2)
#chain, logpost_values = run_PSGLD_training(p0, 50000;
#    a=1e-2, b=0.85, γ=1e-2)
#subset_chain = MCMCChains.subset(chain, 2500:size(chain)[1])
#plot_chain_vs_true(subset_chain, ptrue)

#==============================================================================#
# Some of the more standard Turing sampling APIs require a Turing @model
# macro to be defined. This is illustrated here.
#==============================================================================#

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
iterations = 100000
ϵ = 0.05
τ = 10

chain = sample(target(data_vectorized), HMC(ϵ,τ), iterations, progress=true)
subset_chain = MCMCChains.subset(chain, 2500:size(chain)[1])
plot_chain_vs_true(chain, ptrue)

#==============================================================================#
# Chain diagnostics/summary stats
# These are documented here:https://turinglang.github.io/MCMCChains.jl/dev/stats/
#==============================================================================#
display(summarystats(subset_chain))
display(mean(subset_chain))
display(quantile(subset_chain, q=[0.05, 0.95]))

#==============================================================================#
# Posterior predictive sampling
# https://turing.ml/dev/docs/using-turing/guide#sampling-from-a-conditional-distribution-the-posterior
#==============================================================================#
# Doing this transform of predict() output because MCMCChains produces an
# annoying timestamp error otherwise.
scN = size(subset_chain)[1]
N_post_samples = 100
postp_chain = MCMCChains.subset( subset_chain, scN-N_post_samples:scN)
postp_samples = posterior_predictive_samples =
    get_MCMCChain_from_raw_samples(get_raw_samples_from_MCMCChain(
    predict(predictive_target, subset_chain)))

display(mean(postp_samples))
display(quantile(postp_samples; q=[0.05, 0.95]))
display(summarystats(postp_samples))

plot_predictive_samples(postp_samples)
plot_predictive_quantiles(postp_samples)
plot_data!()
