"""
    allocate_V_and_Q([T], mdp)

Create an empty matrix of state values `V` and state-action values `Q`, 
given a finite Markov decission problem, and an optional type `T`.

Elements of the created matrices are of type `T`. If the type is not specified
`Float64` is used by default.

The content of the created matrices is uninitialized.
"""
function allocate_V_and_Q(T::Type, mdp::FiniteMDP)
    V = Vector{T}(undef, states_no(mdp))
    Q = Matrix{T}(undef, states_no(mdp), actions_no(mdp))
    return V, Q
end

allocate_V_and_Q(mdp::FiniteMDP) = allocate_V_and_Q(Float64, mdp)

"""
    Q_from_V!(Q, V, mdp, γ)

Reevaluate matrix of state-action values `Q` given a vector of state values `V`
for the given finite MDP, assuming discount factor `γ`.
"""
function Q_from_V!(Q::AbstractMatrix{<:Real}, V::AbstractVector{<:Real}, mdp::StochasticFiniteMDP{<:Real}, γ::Real)
    for s = 1:states_no(mdp)
        for a = 1:actions_no(mdp)
            ℙsr = mdp.probabilities(s, a)
            Q[s, a] = 0
            for s_next = 1:states_no(mdp)
                for r in mdp.rewards
                    Q[s, a] += ℙsr[s_next, r] * (r + γ * V[s_next])
                end # for: rewards
            end # for: next states
        end # for: actions
    end # for: states
end

function Q_from_V!(Q::AbstractMatrix{<:Real}, V::AbstractVector{<:Real}, mdp::DeterministicFiniteMDP, γ::Real)
    for s = 1:states_no(mdp)
        for a = 1:actions_no(mdp)
            Q[s, a] = mdp.reward(s, a) + γ * V[mdp.next_state(s, a)]
        end
    end
end

"""
    V_from_Q!(V, Q, policy)

Reevaluate vector of state values `V` given a vector of state-action values `Q`
and a decision policy `policy`.

If `policy` is a matrix, it will be interpreted as a stochastic decision policy,
with entry at coordinate `policy[s,a]` to be interpreted as the probability of choosing
action `a` at state `s`.

If `policy` is a vector, it will be interpreted as a deterministic decision policy,
with entry at coordinate `policy[s]` to be interpreted as the action to choose at state `s`.
"""
function V_from_Q!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real}, stochastic_policy::AbstractMatrix{<:Real})
    Δ = -Inf
    for s = 1:size(Q, 1)
        prev = V[s]
        V[s] = 0
        for a = 1:size(Q, 2)
            V[s] += stochastic_policy[s, a] * Q[s, a]
        end # for: actions
        Δ = max(Δ, abs(V[s] - prev))
    end # for: states
    return Δ
end

function V_from_Q!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real}, deterministic_policy::AbstractVector{<:Integer})
    Δ = -Inf
    for s = 1:size(Q, 1)
        prev = V[s]
        V[s] = Q[s, deterministic_policy[s]]
        Δ = max(Δ, abs(V[s] - prev))
    end # for: states
    return Δ
end

"""
    create_P_from_Q(Q)

Create a vector representing deterministic decission policy based on the size
of the given matrix of state-action values `Q`.

The content of the created vector is uninitialized! After creation, the content
is completely unrelated to the content of `Q`.
"""
function create_P_from_Q(T::Type, Q::AbstractMatrix{<:Real})
    return Vector{T}(undef, size(Q, 1))
end

create_P_from_Q(Q::AbstractMatrix{<:Real}) = create_P_from_Q(Int, Q)

"""
    P_from_Q(Q)

Evaluates deterministic greedy policy (represented by a vector whose entry at index `s`
corresponds to the action to take in `s`) from the given state-action matrix `Q`.

Returns boolean indicator showing whether policy has changed or not.
"""
function P_from_Q!(P::AbstractVector{<:Integer}, Q::AbstractMatrix{<:Real})
    modified = false
    for s = 1:size(Q, 1)
        temp = P[s]
        q = replace(Q[s, :], NaN => -Inf)
        P[s] = argmax(q)
        if isnan(P[s])
            P[s] = rand(1:size(Q, 2))
        end
        if temp != P[s]
            modified = true
        end
    end # for: states
    return modified
end
