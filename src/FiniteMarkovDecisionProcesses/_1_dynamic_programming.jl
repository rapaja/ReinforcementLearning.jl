"""
    policy_evaluation!(V, Q, mdp, policy, γ; tol, maxiter)

Evaluates (deterministic or stochastic) decision policy `policy` for the given Markov
decision process `mdp` and given discount factor `γ`.

By saying that a policy has been "evaluated", we imply that state values `V` and 
state-action values `Q` are computed for this policy. This function evaluates policies in-place,
approximately, in an iterrative manner, until maximal difference between state values in consequtive 
iterations is less than the specified `tol`, or until a maximal number of iterations `maxiter` 
is performed.

The function returns a 2-tuple containing: the number of iterations performed, and a boolean 
indicator of convergence (which is `true` if the vector of state values is evaluated with the 
required precission, and `false` otherwise).
"""
function policy_evaluation!(V::AbstractVector{T}, Q::AbstractMatrix{T}, mdp::FiniteMDP, policy, γ::T; tol = 1e-2, maxiter = 100) where {T<:Real}
    for i = 1:maxiter
        Q_from_V!(Q, V, mdp, γ)
        Δ = V_from_Q!(V, Q, policy)
        if Δ < tol
            return i, true
        end
    end # for: iterations
    return maxiter, false
end

"""
policy_iteration(P, mdp, γ; tol, maxiter, policy_evaluation_maxiter)

Evaluates optimal deterministic decision policy vector `P` for the given Markov
decision process `mdp` and given discount factor `γ`.

The function returns a 2-tuple containing: the number of iterations performed, and a boolean 
indicator of convergence (which is `true` if the vector of state values is evaluated with the 
required precission, and `false` otherwise).
"""
function policy_iteration!(P::AbstractVector{Integer}, mdp::FiniteMDP, γ::T; tol = 1e-2, maxiter = 100, policy_evaluation_maxiter = 5) where {T<:Real}
    V = Vector{T}(length(P))
    Q = create_Q_from_V!(V)
    for i = 1:maxiter
        policy_evaluation!(V, Q, mdp, P, γ; tol = tol, maxiter = policy_evaluation_maxiter)
        modified = P_from_Q!(P, Q)
        if !modified
            return i, true
        end
    end # for: iterations
    return maxiter, false
end