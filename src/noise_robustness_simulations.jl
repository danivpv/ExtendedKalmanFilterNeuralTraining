using DrWatson
using Dictionaries, CSV
using Statistics: mean
using Distributions: MvNormal
using LinearAlgebra: Diagonal

function MSE(pred, data)
    @assert names(pred) == names(data) && nrow(pred) == nrow(data)
    sts_vars = names(data)
    N = nrow(data)
    SE = Dictionary(Dict((var => 0.0 for var in sts_vars)))
    for i in 1:nrow(pred)
        for var in sts_vars
            SE[var] += (pred[i,var] - data[i,var])^2
        end
    end
    return map(x -> x / N, SE)
end

function MRE(pred, data)
    @assert names(pred) == names(data) && nrow(pred) == nrow(data)
    sts_vars = names(data)
    N = nrow(data)
    RE = Dictionary(Dict(var => 0.0 for var in sts_vars))
    for i in 1:nrow(pred)
        for var in sts_vars
            RE[var] += abs((pred[i,var] - data[i,var])/ (data[i,var] + eps()) )
        end
    end
    return map(x -> x / N, RE)
end

function RSE(pred, data)
    @assert names(pred) == names(data) && nrow(pred) == nrow(data)
    sts_vars = names(data)
    SE = Dict(var => 0.0 for var in sts_vars)
    Variation = Dict(var => 0.0 for var in sts_vars)
    means = Dict(var => mean(data[!,var]) for var in sts_vars)
    for i in 1:nrow(pred)
        for var in sts_vars
            SE[var] += (pred[i,var] - data[i,var])^2
            Variation[var] += (pred[i,var] - means[var])^2
        end
    end
    return SE / Variation
end

gaussian_noise(σ) = (col -> (rand(MvNormal(col, σ^2 * Diagonal(col .^ 2)))));
uniform_noise(ϵ) = (col -> col + ϵ * col  .* (rand(Float64, size(col)) * 2 .- 1))

dBs2ϵ(x) = sqrt(3*10^(-x/10))
dBs2σ(x) = sqrt(10^(-x/10))

DataSchema() = DataFrame(σ = Float64[], MSE = Float64[], MRE = Float64[], RSE = Float64[], reference = String[])

function noise_robustness_simulations(N, true_data, algorithm, noise_func, noise_parameter, name, sts; path="noise_robustness")
    means = Dict(var => DataSchema() for var in sts)
    for n in 1:N
        sample = Dict(var => DataSchema() for var in sts)
        for σ in noise_parameter
            noisy_x = mapcols(noise_func(σ), true_data);
            RHONN, params = HIV_model(noisy_x)
            X = train!(RHONN, noisy_x, params; algorithm=algorithm);

            metrics = Dict(
                "ground_truth" => Dict(
                    "MSE" => MSE(true_data, X), 
                    "MRE" => MRE(true_data, X), 
                    "RSE" => RSE(true_data, X)
                ),
                "noisy_data" => Dict(
                    "MSE" => MSE(noisy_x, X), 
                    "MRE" => MRE(noisy_x, X), 
                    "RSE" => RSE(noisy_x, X)
                )
            )

            row = map(
                metrics, 
                m -> Dict(var => Dict(
                    "σ" => σ, 
                    "MSE" => m["MSE"][var],
                    "MRE" => m["MRE"][var], 
                    "RSE" => m["RSE"][var]
                )
                for var in keys(m["RSE"]))
            )

            preprocessed_row = Dict(var => vcat(
                    hcat(DataFrame(row["ground_truth"][var]), DataFrame(reference = 0.)),
                    hcat(DataFrame(row["noisy_data"][var]), DataFrame(reference = 1.))
                ) for var in sts)


            for var in sts
                sample[var] = vcat(sample[var], preprocessed_row[var])
            end

        end
        if n > 1
            for var in sts
                means[var] += sample[var]
            end
        elseif n == 1
            means = sample
        end
    end
    
    for var in sts
        means[var] ./= N
    end

    export_df_dict(means, datadir(path, name))
end

using XLSX

get_sims_data(name) = Dict(first(split(file, ".")) => begin
            DataFrame(XLSX.readtable(joinpath(datadir(name), file), 1)...) 
end for file in readdir(datadir(name)))

function merge_sims(path, ID)
    files = readdir(path)
    sims_data = Dict(alg => get_sims_data(joinpath(path, alg)) for alg in files);
    f = first(keys(sims_data))
    for alg in keys(sims_data), var in keys(sims_data[f])
        sims_data[alg][var][:, ID] .= [alg for i in nrow(sims_data[alg][var])] 
    end
    return Dict(var => begin
        vcat([sims_data[alg][var] for alg in files]...)
    end for var in keys(sims_data[f]))
end

function export_df(x, name, path)
    mkpath(path) 
    XLSX.writetable(joinpath(path, name*".xlsx"), x; overwrite=true)
end

function export_df_dict(x, path)
    mkpath(path) 
    for var in keys(x)
        export_df(x[var], var, path)
    end
end

import_df(path) = DataFrame(XLSX.readtable(path, 1)...)

import Base: -,+, /

+(x::DataFrame, y::DataFrame) = begin
    inner_join_names = intersect(names(x), names(y))
    @assert length(inner_join_names) > 0
    DataFrame(Matrix(x[!,inner_join_names]) .+ Matrix(y[!,inner_join_names]), inner_join_names);
end

+(x::String, y::String) = (@assert x == y; return x)

-(x::DataFrame, y::DataFrame) = begin
    inner_join_names = intersect(names(x), names(y))
    @assert length(inner_join_names) > 0
    DataFrame(Matrix(x[!,inner_join_names]) .- Matrix(y[!,inner_join_names]), inner_join_names);
end

/(x::DataFrame, y::DataFrame) = begin
    inner_join_names = intersect(names(x), names(y))
    @assert length(inner_join_names) > 0
    DataFrame(Matrix(x[!,inner_join_names]) ./ Matrix(y[!,inner_join_names]), inner_join_names);
end

/(x::Dict, y::Dict) = begin
    inner_join_keys = intersect(keys(x), keys(y))
    @assert length(inner_join_keys) > 0
    Dict(k => (x[k] / y[k]) for k in inner_join_keys)
end

import Base: map

map(d::Dict, f::Function) = begin
    Dict(var => f(d[var]) for var in keys(d))
end