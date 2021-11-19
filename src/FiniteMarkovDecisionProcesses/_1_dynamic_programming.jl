function policy_evaluation_V(mdp::StochasticFiniteMDP{T}, policy, γ::T, V0::Vector{T}; tol = 1e-2, maxiter = 100) where {T}
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

function policy_iteration_V(mdp::StochasticFiniteMDP, policy0::Function; policy_evaluation_iters_no = 5)

end