using DrWatson
@quickactivate :ExtendedKalmanFilterNeuralTraining
#@quickactivate "ExtendedKalmanFilterNeuralTraining"; include(srcdir("ExtendedKalmanFilterNeuralTraining.jl"))

using XLSX, DataFrames

using LinearAlgebra: Diagonal
using Distributions: MvNormal

using ModelingToolkit, DifferentialEquations

@named hiv = HIV()
display(hiv)

HIV_tspan = (0.0, 365.0 * 10)
HIV_prob = ODEProblem(hiv, [], HIV_tspan)
HIV_sol = solve(HIV_prob);

x = DataFrame(hcat(map(HIV_sol, 0.0:2:max(HIV_tspan...))...)', ["T", "T_inf", "M", "M_inf", "V"]);

SST = "Tikonov_Sample_Metrics"
SSEFK = "EFK_Sample_Metrics"
N = 5
dB = 20
σ = dBs2σ.(dB)
noise_robustness_simulations(N, x, TikonovEKF!(α=5e15), gaussian_noise, [σ], SST, names(x))
noise_robustness_simulations(N, x, EKF!(), gaussian_noise, [σ], SSEFK, names(x))

Tikonov_Sample_Metrics = get_sims_data(joinpath("noise_robustness",SST));
EFK_Sample_Metrics = get_sims_data(joinpath("noise_robustness",SSEFK));

RHONN, params = HIV_model(gaussian_noisy_x)
X_Tikonov = train!(RHONN, gaussian_noisy_x, params; algorithm=TikonovEKF!(α=5e15));

export_df(X_Tikonov, "TikonovEFK", projectdir("noise_robustness", "Single_Event"))
export_df(x, "data", projectdir("noise_robustness", "Single_Event"))
export_df(gaussian_noisy_x, "noisy_data", projectdir("noise_robustness", "Single_Event"))