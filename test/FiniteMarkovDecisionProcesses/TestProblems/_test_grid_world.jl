@testset "GridWorld tests" begin

    next_states = [
        1 5 2 15
        2 6 3 1
        3 7 3 2
        15 8 5 4
        1 9 6 4
        2 10 7 5
        3 11 7 6
        4 12 9 8
        5 13 10 8
        6 14 11 9
        7 15 11 10
        8 12 13 12
        9 13 14 12
        10 14 15 13
        15 15 15 15
    ]

    rewards = -1 * ones(size(next_states))
    rewards[15, :] .= 0

    fmdp = MDP.DeterministicFiniteMDP((s, a) -> next_states[s, a], (s, a) -> rewards[s, a], 4, 15, 15)

    optimal_V = [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -1.0, -3.0, -2.0, -1.0, 0.0]

    uniform_random_policy = 0.25 * ones(size(next_states)...)
    V_uniform_random_policy = [-14.0, -20.0, -22.0, -14.0, -18.0, -20.0, -20.0, -20.0, -20.0, -18.0, -14.0, -22.0, -20.0, -14.0, 0.0]

    random_deterministic_policy = rand(1:4, size(next_states, 1))

    @testset "Dynamic Programming (DP) tests" begin

        dp_V_tol = 1e-4
        dp_maxiter_eval = 1000
        dp_maxiter_optim = 1000

        @testset "DP: policy evaluation (textbook)" begin
            𝐏 = copy(uniform_random_policy)

            V = MDP.allocate_V(fmdp)
            Q = MDP.allocate_Q(fmdp)

            V[MDP.terminal_state(fmdp)] = 0
            iters_no, converged = MDP.dp_evaluate_policy_textbook!(V, Q, fmdp, 𝐏, 1.0; tol = dp_V_tol, maxiter = dp_maxiter_eval)

            ΔV = abs.((V - V_uniform_random_policy))
            err = max(ΔV...)
            @info "DP: policy evaluation (textbook): max abs err = $err (iters no = $iters_no)"
            @test converged && isapprox(err, 0.0; atol = 1e-1)
        end

        @testset "DP: policy evaluation via V update" begin
            𝐏 = copy(uniform_random_policy)

            V = MDP.allocate_V(fmdp)
            V[MDP.terminal_state(fmdp)] = 0

            converged = false
            iters_no = dp_maxiter_eval
            for i = 1:dp_maxiter_eval
                Δ = MDP.dp_update_V!(V, fmdp, 𝐏, 1.0)
                if Δ < dp_V_tol
                    converged = true
                    iters_no = i
                    break
                end
            end

            ΔV = abs.((V - V_uniform_random_policy))
            err = max(ΔV...)
            @info "DP: policy evaluation (V update): max abs err = $err (iters no = $iters_no)"
            @test converged && isapprox(err, 0.0; atol = 1e-1)
        end

        @testset "DP: policy optimization (textbook GPI)" begin
            𝐩 = copy(random_deterministic_policy)

            V = MDP.allocate_V(fmdp)
            Q = MDP.allocate_Q(fmdp)

            V[MDP.terminal_state(fmdp)] = 0
            converged = false
            iters_no = dp_maxiter_optim
            for i = 1:dp_maxiter_optim
                MDP.dp_evaluate_policy_textbook!(V, Q, fmdp, 𝐩, 1.0; tol = dp_V_tol, maxiter = 5)
                modified = MDP.𝐩_from_Q!(𝐩, Q)
                if !modified
                    iters_no = i
                    converged = true
                    break
                end
            end # for: iterations

            ΔV = abs.((V - optimal_V))
            err = max(ΔV...)
            @info "DP: policy optimization (textbook GPI): max abs err = $err (iters no = $iters_no)"
            @test converged && isapprox(err, 0.0; atol = 1e-1)
        end

        @testset "DP: policy optimization (value iteration)" begin
            V = MDP.allocate_V(fmdp)
            V[MDP.terminal_state(fmdp)] = 0

            converged = false
            iters_no = dp_maxiter_optim
            for i = 1:dp_maxiter_optim
                Δ = MDP.dp_update_V!(V, fmdp, 1.0)
                if Δ < dp_V_tol
                    converged = true
                    iters_no = i
                    break
                end
            end

            ΔV = abs.((V - optimal_V))
            err = max(ΔV...)
            @info "DP: policy optimization (value iteration): max abs err = $err (iters no = $iters_no)"
            @test converged && isapprox(err, 0.0; atol = 1e-1)
        end

    end

    @testset "Monte Karlo (MK) tests" begin

        @testset "MK: policy evaluation" begin
            𝐏 = copy(uniform_random_policy)
            simulator = MDP.create_simulator(fmdp, 𝐏, 1000)

            V = MDP.allocate_V(fmdp)
            Q = MDP.allocate_Q(fmdp)

            MDP.mk_evaluate_policy!(Q, simulator, 1.0; maxiter = 10000)

            MDP.V_from_Q!(V, Q, 𝐏)
            ΔV = abs.((V - V_uniform_random_policy))
            err = max(ΔV...)
            @info "MK: policy evaluation: max abs err = $err"
            @test isapprox(err, 0.0; atol = 1)
        end

        @testset "MK: ε-greedy simulation" begin
            Q = MDP.allocate_Q(fmdp)
            MDP.Q_from_V!(Q, optimal_V, fmdp, 1.0)

            𝐩 = copy(random_deterministic_policy)
            MDP.𝐩_from_Q!(𝐩, Q)

            simulator = MDP.create_simulator(fmdp, 𝐩, 0.05, 100)
            episode = simulator(5, 2)
        end

        @testset "MK: policy optimization" begin
            𝐩 = copy(random_deterministic_policy)
            simulator = MDP.create_simulator(fmdp, 𝐩, 0.05, 100)

            Q = MDP.allocate_Q(fmdp)
            MDP.𝐩_from_Q!(𝐩, Q)

            for i = 1:10
                MDP.mk_evaluate_policy!(Q, simulator, 1.0; maxiter = 10000)
                modified = MDP.𝐩_from_Q!(𝐩, Q)
            end # for: iterations

            V = MDP.allocate_V(fmdp)
            MDP.V_from_Q!(V, Q, 𝐩)
            ΔV = abs.(V - optimal_V)
            err = max(ΔV...)
            @info "MK: policy optimization: max abs err = $err"
            @test isapprox(err, 0.0; atol = 1)
        end

    end

end