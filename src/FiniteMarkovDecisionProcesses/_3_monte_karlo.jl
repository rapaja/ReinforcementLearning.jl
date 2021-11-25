#TODO: Some kind of generator/iterator/resumable function would be prefered here...
function read_return(episode, i, γ, non_termination_penalty)
    s, a, r = episode[i]
    g = r
    w = 1
    for j = (i+1):length(episode)-1
        w = w * γ
        _, _, r1 = episode[j]
        g += w * r1
    end
    s_fin, _, _ = episode[end]
    if s_fin != TERMINAL_STATE_CODE
        g -= non_termination_penalty
    end
    return s, a, g
end

function update_state_action_values_repository!(R, episode, γ::Real, non_termination_penalty)
    for i = 1:length(episode)-1
        s, a, g = read_return(episode, i, γ, non_termination_penalty)
        ∑r, n = R[s, a]
        R[s, a] = (∑r + g, n + 1)
    end
end

function Q_from_state_action_values_repository!(Q, R)
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
    for _ = 1:maxiter
        s0 = rand(1:size(Q, 1))
        a0 = rand(1:size(Q, 2))
        episode = policy_simulator(s0, a0)
        update_state_action_values_repository!(R, episode, γ, non_termination_penalty)
    end # for: iterations
    Q_from_state_action_values_repository!(Q, R)
end

function mk_update_Q!(Q, ω, policy_simulator, γ; non_termination_penalty = 1000)
    s0 = rand(1:size(Q, 1))
    a0 = rand(1:size(Q, 2))
    episode = policy_simulator(s0, a0)
    Δ = -Inf
    for i = 1:length(episode)-1
        s, a, g = read_return(episode, i, γ, non_termination_penalty)
        new_value = ω * Q[s, a] + (1 - ω) * g
        Δ = max(Δ, abs(new_value - Q[s, a]))
        Q[s, a] = new_value
    end
    return Δ
end

