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
    wsave(datadir("EKFvsTikonovEKF", savename(d, "jld2")), f)
end

EKFvsTikonovEKF = collect_results(datadir("EKFvsTikonovEKF"))

XLSX.writetable(datadir("EKFvsTikonovEKF", "full_table.xlsx"), EKFvsTikonovEKF; overwrite=true)

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
    wsave(datadir("Robustness", savename(d, "jld2")), f)
end

Robustness = collect_results(datadir("Robustness"))

XLSX.writetable(datadir("Robustness", "full_table.xlsx"), Robustness, overwrite=true)


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
    wsave(datadir("NoiseTypeComparison", savename(d, "jld2")), f)
end

NoiseTypeComparison = collect_results(datadir("NoiseTypeComparison"))

XLSX.writetable(datadir("NoiseTypeComparison", "full_table.xlsx"), NoiseTypeComparison, overwrite=true)

### Set permissions of created dirs to public

chmod("data", 0o777; recursive=true)