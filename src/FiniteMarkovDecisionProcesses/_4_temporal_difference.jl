#TODO: Some kind of iterator/resumable would be preferable
function read_SARSA_sequence(episode)
    SARSA_sequence = Vector{Tuple{Int,Int,Float64,Int,Int}}()
    n = length(episode)
    if n > 1
        s, a, r = episode[1]
        # Skip last entry, because it is a terminal state
        for i = 2:n-1
            sn, an, rn = episode[i]
            push!(SARSA_sequence, (s, a, r, sn, an))
            s, a, r = sn, an, rn
        end
    end
    return SARSA_sequence
end

function td_0!(V, ω, r, s, sn, γ)
    ΔV = ω * (r + γ * V[sn] - V[s])
    V[s] = V[s] + ΔV
    return ΔV
end