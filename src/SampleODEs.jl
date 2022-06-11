using ModelingToolkit, DiffEqBase

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

# Cancer


# 