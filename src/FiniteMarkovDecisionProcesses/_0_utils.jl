"""
    create_Q_from_V(V, actions_no)

Create an empty matrix of state-action values `Q`, given a vector of state values `V`
and the total number of actions `actions_no`.

The content of the created matrix is uninitialized! After creation, the content
is completely unrelated to the content of `V`.
"""
function create_Q_from_V(V::AbstractVector{<:Real}, actions_no::Integer)
    return Matrix{eltype(V)}(undef, length(V), actions_no)
end

"""
    Q_from_V!(Q, V, mdp, γ)

Reevaluate matrix of state-action values `Q` given a vector of state values `V`
for the given finite MDP, assuming discount factor `γ`.
"""
function Q_from_V!(Q::AbstractMatrix{T}, V::AbstractVector{T}, mdp::StochasticFiniteMDP{T}, γ::T) where {T<:Real}
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

function Q_from_V!(Q::AbstractMatrix{T}, V::AbstractVector{T}, mdp::DeterministicFiniteMDP, γ::T) where {T<:Real}
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
function V_from_Q!(V::AbstractVector{T}, Q::AbstractMatrix{T}, stochastic_policy::AbstractMatrix{T}) where {T<:Real}
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

function V_from_Q!(V::AbstractVector{T}, Q::AbstractMatrix{T}, deterministic_policy::AbstractVector{Integer}) where {T<:Real}
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
function create_P_from_Q(Q::AbstractMatrix{T}) where {T<:Real}
    return Vector{T}(undef, size(Q, 1))
end

"""
    P_from_Q(Q)

Evaluates deterministic greedy policy (represented by a vector whose entry at index `s`
corresponds to the action to take in `s`) from the given state-action matrix `Q`.

Returns boolean indicator showing whether policy has changed or not.
"""
function P_from_Q!(P::AbstractVector{Integer}, Q::AbstractMatrix{T}) where {T<:Real}
    modified = false
    for s = 1:size(Q, 1)
        temp = P[s]
        P[s] = argmax(Q[s, :])
        if temp != P[s]
            modified = true
        end
    end # for: states
    return modified
end
