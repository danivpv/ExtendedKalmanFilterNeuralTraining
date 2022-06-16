using ModelingToolkit, DiffEqBase
@variables t 
d = Differential(t) 

# Viruses
function IAV(
    E0=1.0e6, V0=25.0, D0=0.0, η0=0.0, 
    k_v=1.0e6, p=4.4, c_v=1.24e-6, 
    S_E=2.0e-2 * 1.0e6, c_e=2.0e-2, r=0.33, k_e=2.7e3, 
    δ_d=3.26, EC_50=42.3; 
    name
    )

    ps = @parameters k_v=k_v p=p c_v=c_v S_E=S_E c_e=c_e r=r K_e=k_e δ_d=δ_d EC_50=EC_50 
    sts = @variables E(t)=E0 V(t)=V0 D(t)=D0 η(t)=η0
    
    eqs = [
        d(E) ~ S_E + r * E * (V / (V + k_e)) - c_e * E,
        η ~ D / (D + EC_50),
        d(V) ~ p * (1 - η) * V * (1 - (V / k_v)) - c_v * V * E,
        d(D) ~ -δ_d * D
        ]
    
    ODESystem(eqs, t, sts, ps; name)
end

function HIV(
    T0=1.0e3, T0_inf=0.0, M0=150.0, M0_inf=0.0, V0=10.0,
    ρ_T=0.01, C_T=300, k_T=4.57e-5, s_T=10.0, p_T=38.0,
    δ_Tinf=0.4, 
    ρ_M=0.003, C_M=200, δ_T=0.01, s_M=0.15, k_M=4.33e-8, δ_M=1.0e-3, p_M=35.0,
    δ_Minf=1.0e-3,
    δ_V=2.4;
    name
    )

    ps = @parameters ρ_T=ρ_T C_T=C_T k_T=k_T s_T=s_T p_T=p_T δ_Tinf=δ_Tinf ρ_M=ρ_M C_M=C_M δ_T=δ_T s_M=s_M k_M=k_M δ_M=δ_M p_M=p_M δ_Minf=δ_Minf δ_V=δ_V
    sts = @variables T(t)=T0 T_inf(t)=T0_inf M(t)=M0 M_inf(t)=M0_inf V(t)=V0
    
    eqs = [
        d(T) ~ s_T + ρ_T * V * T / (C_T + V) - k_T * T * V - δ_T * T,
        d(T_inf) ~ k_T * T * V - δ_Tinf * T_inf,
        d(M) ~ s_M + ρ_M * V * M / (C_M + V) - k_M * M * V - δ_M * M,
        d(M_inf) ~ k_M * M * V - δ_Minf * M_inf,
        d(V) ~ p_T * T_inf + p_M * M_inf - δ_V * V
        ]
    
    ODESystem(eqs, t, sts, ps; name)
end

using LinearAlgebra: I, dot

function HIV_model(x)
    T = Dict{String,Any}()
    T_inf = Dict{String,Any}()
    M = Dict{String,Any}()
    M_inf = Dict{String,Any}()
    V = Dict{String,Any}()
    params = Dict{String,Any}()

    Q = Matrix(1.0e6I, 2, 2)
    R = 1.0e4
    η = 1.0
    P = Matrix(10.0I, 2, 2) 
    ω = rand(2)*1200 
    @pack! T = Q, R, η, P, ω

    Q = Matrix(1.0e6I, 2, 2)
    R = 1.0e4
    η = 1.0
    P = Matrix(10.0I, 2, 2)
    ω = rand(2)
    @pack! T_inf = Q, R, η, P, ω

    Q = Matrix(1.0e6I, 2, 2)
    R = 1.0e4
    η = 1.0
    P = Matrix(10.0I, 2, 2)
    ω = rand(2)
    @pack! M = Q, R, η, P, ω

    Q = Matrix(1.0e6I, 2, 2)
    R = 1.0e4
    η = 1.0
    P = Matrix(10.0I, 2, 2)
    ω = rand(2)*0.5
    @pack! M_inf = Q, R, η, P, ω

    Q = Matrix(1.0e6I, 3, 3)
    R = 1.0e4
    η = 2.0
    P = Matrix(10.0I, 3, 3)
    ω = rand(3)*2
    @pack! V = Q, R, η, P, ω

    params = Dict{String,Any}()
    @pack! params = T, T_inf, M, M_inf, V

    ϕ(v) = 1 / (1 + exp(-v))
    RHONN(k) = Dict(
        "T" => (ω -> [dot(ω, [ϕ(x[k,"T"]), ϕ(x[k,"T"]) * ϕ(x[k,"V"])])]), 
        "T_inf" => (ω -> [dot(ω, [ϕ(x[k,"T"]) * ϕ(x[k,"V"]), ϕ(x[k,"T_inf"])])]), 
        "M" => (ω -> [dot(ω, [ϕ(x[k,"M"]) * ϕ(x[k,"V"]), ϕ(x[k,"V"])])]),
        "M_inf" => (ω -> [dot(ω, [ϕ(x[k,"M_inf"]) * ϕ(x[k,"V"]), ϕ(x[k,"M_inf"])])]),
        "V" => (ω -> [dot(ω, [ϕ(x[k,"T_inf"]), ϕ(x[k,"M_inf"]), ϕ(x[k,"V"])])])
        )

    return RHONN, params
end

function IAV_model(x::DataFrame, u)
    @assert length(u) == nrow(x)
    E = Dict{String,Any}()
    V = Dict{String,Any}()
    D = Dict{String,Any}()
    params = Dict{String,Any}()
    
    Q = Matrix(1.0e6I, 2, 2)
    R = 1.0e4
    η = 0.93
    P = Matrix(10.0I, 2, 2) 
    ω = rand(2)*1200 
    @pack! E = Q, R, η, P, ω

    Q = Matrix(1.0e6I, 3, 3)
    R = 1.0e4
    η = 0.63
    P = Matrix(1000.0I, 3, 3)
    ω = rand(3)*0.05
    @pack! V = Q, R, η, P, ω

    Q = [1.0e6]
    R = 1.0e4
    η = 0.13
    P = [1.0]
    ω = [rand()*0.05]
    @pack! D = Q, R, η, P, ω

    @pack! params = E, V, D

    ϕ(v) = 1 / (1 + exp(-v))
    RHONN(k) = Dict(
        "E" => (ω -> [dot(ω, [ϕ(val) for val in x[k,["E","V"]]]) + u[k]]), 
        "V" => (ω -> [dot(ω, [ϕ(val) for val in x[k,["E","V","D"]]]) + u[k]]), 
        "D" => (ω -> [ω[1] * ϕ(x[k,"D"]) + u[k]])
        )

    return RHONN, params
end

# Cancer


# 