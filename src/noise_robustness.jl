using DrWatson
using Statistics: mean
using Distributions: MvNormal
using LinearAlgebra: Diagonal

gaussian_noise(σ) = (col -> (rand(MvNormal(col, σ^2 * Diagonal(col .^ 2)))))
uniform_noise(ϵ) = (col -> col + ϵ * col  .* (rand(Float64, size(col)) * 2 .- 1))

dBs2ϵ(x) = sqrt(3*10^(-x/10))
dBs2σ(x) = sqrt(10^(-x/10))

algorithms = @strdict EKF! TikonovEKF!
noise_types = @strdict gaussian_noise uniform_noise

function makesim(d::Dict, x; α = 5e15)
    copy_d = deepcopy(d)
    @unpack N, alg, dB, noise = copy_d
    means = noise_robustness(N, x, algorithms[alg](α=α), noise_types[noise], dBs2σ(dB));
    return merge(copy_d, means)
end

function noise_robustness(N, true_data, algorithm, noise_func, noise_parameter)
    sts = names(true_data)
    means = Dict(join([var, m, ref], "_") => 0.0 for var in sts, m in ["MSE", "MRE"] for ref in ["ground_truth", "noisy_data"])
    
    for n in 1:N
        noisy_x = mapcols(noise_func(noise_parameter), true_data);
        RHONN, params = HIV_model(noisy_x)
        X = train!(RHONN, noisy_x, params; algorithm=algorithm);
        
        metrics = Dict()
        for var in sts
            metrics[var*"_MSE_ground_truth"] = norm(true_data[!,var]-X[!,var]) / N
            metrics[var*"_MRE_ground_truth"] = mean(abs.((true_data[!,var]-X[!,var]) ./ true_data[!,var]))
            metrics[var*"_MSE_noisy_data"] = norm(noisy_x[!,var]-X[!,var]) / N
            metrics[var*"_MRE_noisy_data"] = mean(abs.((noisy_x[!,var]-X[!,var]) ./ noisy_x[!,var]))
        end

        means += metrics
    end
    
    for k in keys(means)
        means[k] /= N
    end

    return means
end 


import_df(path) = DataFrame(XLSX.readtable(path, 1)...)

import Base: -,+, /

+(x::Dict, y::Dict) = begin
    inner_join_names = intersect(keys(x), keys(y))
    @assert length(inner_join_names) > 0
    Dict(n => x[n] + y[n] for n in inner_join_names)
end

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