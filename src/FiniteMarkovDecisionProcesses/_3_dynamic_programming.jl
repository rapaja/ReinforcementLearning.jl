"""
    dp_evaluate_policy_textbook!(V, Q, fmdp, policy, γ; tol, maxiter)

Evaluates (deterministic or stochastic) decision policy `policy` for the given finite Markov
decision process `fmdp` and given discount factor `γ`.

The procedure is 'textbook', in the sense that unnecessary allocations and steps have
been made in order to make the procedure more explicit and readable.

By saying that a policy has been "evaluated", we imply that state values `V` and 
state-action values `Q` are computed for this policy. This function evaluates policies in-place,
approximately, in an iterrative manner, until maximal difference between state values in consequtive 
iterations is less than the specified `tol`, or until a maximal number of iterations `maxiter` 
is performed.

The function returns a 2-tuple containing: the number of iterations performed, and a boolean 
indicator of convergence (which is `true` if the vector of state values is evaluated with the 
required precission, and `false` otherwise).
"""
function dp_evaluate_policy_textbook!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real}, fmdp::FiniteMDP, policy, γ::Real; tol = 1e-2, maxiter = 100)
    for i = 1:maxiter
        Q_from_V!(Q, V, fmdp, γ)
        Δ = V_from_Q!(V, Q, policy)
        if Δ < tol
            return i, true
        end
    end # for: iterations
    return maxiter, false
end

"""
    dp_update_V!(V, fmdp, [policy,] γ::Real)

Performs one-step in-place update of the vector of state values `V` for the given finite Markov Decision Process
`fmdp` (deterministic or stochastic) and given discount factor `γ`.

If decision policy `policy` is given, the update will be perform for that particular policy. Otherwise, the
update will be performed for the deterministic greedy policy with respect to the current state values.

Return the maximal absolute update of any element of `V`.
"""
function dp_update_V!(V::AbstractVector{<:Real}, fmdp::FiniteMDP, policy, γ::Real)
    Δ = -Inf
    for s = 1:states_no(fmdp)
        new_value = expected_return(fmdp, s, policy, V, γ)
        Δ = max(Δ, abs(V[s] - new_value))
        V[s] = new_value
    end # for: states
    return Δ
end

function dp_update_V!(V::AbstractVector{<:Real}, fmdp::FiniteMDP, γ::Real)
    Δ = -Inf
    for s = 1:states_no(fmdp)
        new_value = -Inf
        for a = 1:actions_no(fmdp)
            new_value = max(new_value, expected_return(fmdp, s, a, V, γ))
        end # for: actions
        Δ = max(Δ, abs(V[s] - new_value))
        V[s] = new_value
    end # for: states
    return Δ
end