function td_0_update_V!(V, ω, s, r, sn, γ)
    ΔV = ω * (r + γ * V[sn] - V[s])
    V[s] = V[s] + ΔV
    return ΔV
end

function td_sarsa_update_Q!(Q, ω, s, a, r, sn, an, γ)
    ΔQ = ω * (r + γ * Q[sn, an] - Q[s, a])
    Q[s, a] = Q[s, a] + ΔQ
    return ΔQ
end

function td_ql_update_Q!(Q, ω, s, a, r, sn, γ)
    Qmax = max(Q[sn, :]...)
    ΔQ = ω * (r + γ * Qmax - Q[s, a])
    Q[s, a] = Q[s, a] + ΔQ
    return ΔQ
end