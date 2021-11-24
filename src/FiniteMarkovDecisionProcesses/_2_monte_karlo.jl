"""
    mk_evaluate_policy!(V, Q, mdp, policy, γ; tol, maxiter)

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
function mk_evaluate_policy!(Q::AbstractMatrix{<:Real}, policy_simulator, γ::Real; maxiter = 100, non_termination_penalty = 1000)
    R = fill((0.0, 0), size(Q))
    for i = 1:maxiter
        s0 = rand(1:size(Q, 1))
        a0 = rand(1:size(Q, 2))
        episode = policy_simulator(s0, a0)
        for i = 1:length(episode)-1
            s, a, r = episode[i]
            g = r
            w = 1
            for j = (i+1):length(episode)-1
                w = w * γ
                _, _, r1 = episode[j]
                g += w * r1
            end
            s_fin, _, _ = episode[end]
            if s_fin != -1
                g -= non_termination_penalty
            end
            ∑r, n = R[s, a]
            R[s, a] = (∑r + g, n + 1)
        end
    end # for: iterations
    for s = 1:size(Q, 1)
        for a = 1:size(Q, 2)
            ∑r, n = R[s, a]
            if n != 0
                Q[s, a] = ∑r / n
            else
                Q[s, a] = NaN
            end
        end
    end
end
