module ExtendedKalmanFilterNeuralTraining

export train!
export EKF!, TikonovEKF!

# Dependencies
using Reexport, UnPack, ForwardDiff
@reexport using Revise, DifferentialEquations, DataFrames, LinearAlgebra, Statistics, Random
 # Reexport, UnPack, ForwardDiff, Dictionaries, Revise, DifferentialEquations, Plots, LaTeXStrings, XLSX
# Case studies functions with t as independent variable and d as derivative operator 
include("sample_models.jl")
export IAV, HIV, IAV_model, HIV_model

include("plots.jl")
export HIV_plot

include("noise_robustness_simulations.jl")
export noise_robustness_simulations, noise_robustness_simulation, DataSchema, get_sims_data, merge_sims, export_df, import_df, export_df_dict
export +, -, /, map
export gaussian_noise, uniform_noise, dBs2σ, dBs2ϵ, algorithms, noise_types

# Main training cycle
function train!(F, measurements, params; algorithm=EKF!(), save_weights=false)
    sts_vars = sort(names(measurements))
    @assert sort(collect(keys(F(1)))) == sts_vars == sort(collect(keys(params)))

    X = DataFrame(measurements[1,:])
    save_weights ? (W = Dict(var => zeros(Float64, nrow(measurements)) for var in sts_vars)) : nothing
    
    for k in 1:(nrow(measurements)-1)
        f = F(k)

        # Update weights
        for var in sts_vars
            algorithm(measurements[k,var] - X[k,var], params[var], f[var])
            save_weights ? (W[var][k] = norm(params[var]["ω"])) : continue
        end

        # Update state
        push!(X, Dict(var => first(f[var](params[var]["ω"])) for var in sts_vars))
    end
    return save_weights ? (X, DataFrame(W)) : X
end


# Training algorithms

function EKF!(;α=0.0, p_norm=2)
    return (function algorithm(error, params, F) 
        @unpack Q, R, η, P, ω = params

        H = ForwardDiff.jacobian(F, ω)' #requires alloc, gradient might be more efficient
        M = inv(R .+ H' * P * H) # scalar
        K = P * H * M # vector

        # mutates
        ω .+= η * K * (error + α * norm(ω, p_norm))
        P .+= -K * H' * P + Q
    end)
end

function TikonovEKF!(;α=1.0)
    return (function algorithm(error, params, F) 
        @unpack Q, R, η, P, ω = params

        H = ForwardDiff.jacobian(F, ω)' #requires alloc, gradient might be more efficient
        invM = R .+ H' * P * H # scalar
        ReM = inv(invM' * invM + Matrix(α * I, size(invM))) * invM'
        K = P * H * ReM # vector

        # mutates
        ω .+= η * K * error
        P .+= -K * H' * P + Q
    end)
end

algorithms = @strdict EKF! TikonovEKF!
noise_types = @strdict gaussian_noise uniform_noise

end # module
