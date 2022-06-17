using Pkg
Pkg.activate("~/daniel/ExtendedKalmanFilterNeuralTraining")
Pkg.instantiate()

using DrWatson
@quickactivate :ExtendedKalmanFilterNeuralTraining
#@quickactivate "ExtendedKalmanFilterNeuralTraining"; include(srcdir("ExtendedKalmanFilterNeuralTraining.jl"))

using ModelingToolkit, DifferentialEquations

@named hiv = HIV()

HIV_tspan = (0.0, 365.0 * 10)
HIV_prob = ODEProblem(hiv, [], HIV_tspan)
HIV_sol = solve(HIV_prob);

x = DataFrame(hcat(map(HIV_sol, 0.0:2:max(HIV_tspan...))...)', ["T", "T_inf", "M", "M_inf", "V"]);


### Simulation 1

allparams = Dict(
    "N" => 500, 
    "dB" => 20.0,
    "alg" => ["EKF!", "TikonovEKF!"], 
    "noise" => "gaussian_noise"
)

dicts = dict_list(allparams)
for d in dicts
    f = makesim(d, x)
    wsave(datadir("N=$(d["N"])_EKFvsTikonovEKF", savename(d, "jld2")), f)
end

EKFvsTikonovEKF = collect_results(datadir("N=$(d["N"])_EKFvsTikonovEKF"))

XLSX.writetable(datadir("N=$(d["N"])_EKFvsTikonovEKF", "full_table.xlsx"), EKFvsTikonovEKF, overwrite=true)


### Simulation 2

allparams = Dict(
    "N" => 100, 
    "dB" => collect(0:1:20),
    "alg" => "TikonovEKF!",  
    "noise" => "gaussian_noise"
)

dicts = dict_list(allparams)
for d in dicts
    f = makesim(d, x)
    wsave(datadir("N=$(d["N"])_Robustness", savename(d, "jld2")), f)
end

Robustness = collect_results(datadir("N=$(d["N"])_Robustness"))

XLSX.writetable(datadir("N=$(d["N"])_Robustness", "full_table.xlsx"), Robustness, overwrite=true)


### Simulation 3

allparams = Dict(
    "N" => 100, 
    "dB" => collect(0:1:20),
    "alg" => "TikonovEKF!",     
    "noise" => ["gaussian_noise", "uniform_noise"]
)

dicts = dict_list(allparams)
for d in dicts
    f = makesim(d, x)
    wsave(datadir("N=$(d["N"])_NoiseTypeComparison", savename(d, "jld2")), f)
end

NoiseTypeComparison = collect_results(datadir("N=$(d["N"])_NoiseTypeComparison"))

XLSX.writetable(datadir("N=$(d["N"])_NoiseTypeComparison", "full_table.xlsx"), NoiseTypeComparison, overwrite=true)