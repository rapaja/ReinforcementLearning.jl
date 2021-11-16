module Tabular

"""
    init_Q_from_V(V, actions_no)

Initialize an empty matrix of state-action values `Q`, given a vector of state values `V`
and the total number of actiions `actions_no`.
"""
function init_Q_from_V(V::AbstractVector{<:Real}, actions_no::Integer)
    return Matrix{eltype(V)}(undef, length(V), actions_no)
end

function Q_from_V(V::Vector{T}, mdp::FiniteMarkovDecissionProblem{T}, γ::T) where {T<:Real}
    Q = init_Q_from_V(V, mdp.actions_no)
    Q_from_V!(Q, V, mdp, γ)
    return Q
end

function Q_from_V!(Q::Matrix{T}, V::Vector{T}, mdp::FiniteMarkovDecissionProblem{T}, γ::T) where {T<:Real}
    for s = 1:mdp.states_no
        for a = 1:mdp.actions_no
            ℙsr = mdp.probabilities(s, a)
            Q[s, a] = 0
            for s_next = 1:mdp.states_no
                for r in mdp.rewards
                    Q[s, a] += ℙsr[s_next, r] * (r + γ * V[s_next])
                end # for: rewards
            end # for: next states
        end # for: actions
    end # for: states
end

function V_from_Q!(V::Array{T}, Q::Matrix{T}, stochastic_policy::Matrix{T}) where {T<:Real}
    for s = 1:mdp.states_no
        V_next[s] = 0
        for a = 1:mdp.actions_no
            V_next[s] += stochastic_policy[s, a] * Q[s, a]
        end # for: actions
    end # for: states
end

function V_from_Q!(V::Array{T}, Q::Matrix{T}, deterministic_policy::Vector{T}) where {T<:Real}
    for s = 1:mdp.states_no
        V[s] = Q[s, deterministic_policy[s]]
    end # for: states
end


function policy_evaluation_V(mdp::FiniteMarkovDecissionProblem{T}, policy, γ::T, V0::Vector{T}; tol = 1e-2, maxiter = 100) where {T}
    V = copy(V0)
    V_next = similar(V)
    Q = init_Q_from_V(V, mdp.actions_no)
    for i = 1:maxiter
        Q_from_V!(Q, V, mdp, γ)
        V_from_Q(V_next, Q, policy)
        if max.(abs.(V - V_next)) < tol
            return V_next, i, true
        end
        V, V_next = V_next, V  # swap V and V_next
    end # for: iterations
    return V, maxiter, false
end

function deterministic_greedy_policy_from_Q(Q)
    states_no = size(Q)[2]
    actions = zeros(eltype(Q), states_no)
    for s = 1:states_no
        actions[s] = argmax(Q[s, :])
    end # for: states
end

function deterministic_greedy_policy_from_V(mdp::FiniteMarkovDecissionProblem, γ, V)
    Q = Q_from_V(mdp, γ, V)
    return deterministic_greedy_policy_from_Q(Q)
end

function policy_iteration_V(mdp::FiniteMarkovDecissionProblem, policy0::Function; policy_evaluation_iters_no = 5)

end

end # module