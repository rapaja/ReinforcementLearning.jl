function td_0_update_V!(V, ω, r, s, sn, γ)
    ΔV = ω * (r + γ * V[sn] - V[s])
    V[s] = V[s] + ΔV
    return ΔV
end