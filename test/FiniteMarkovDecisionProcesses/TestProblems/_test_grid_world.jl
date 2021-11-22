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

    @testset "DP: policy evaluation" begin
        policy = 0.25 * ones(size(next_states)...)

        V, Q = MDP.allocate_V_and_Q(fmdp)
        V[1] = 0
        iters_no, converged = MDP.dp_evaluate_policy!(V, Q, fmdp, policy, 1.0; tol = 1e-4, maxiter = 1000)

        @test converged == true

        expected_V = [-14.0, -20.0, -22.0, -14.0, -18.0, -20.0, -20.0, -20.0, -20.0, -18.0, -14.0, -22.0, -20.0, -14.0, 0.0]
        @test all(isapprox.(V, expected_V; atol = 1e-2))
    end

    @testset "DP: policy optimization" begin
        P = rand(1:4, size(next_states, 1))
        optimal_V = [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -1.0, -3.0, -2.0, -1.0, 0.0]
        V, Q = MDP.allocate_V_and_Q(fmdp)
        converged = false
        for i = 1:100
            MDP.dp_evaluate_policy!(V, Q, fmdp, P, 1.0; tol = 1e-4, maxiter = 5)
            modified = MDP.P_from_Q!(P, Q)
            if !modified
                converged = true
                break
            end
        end # for: iterations

        @test converged == true
        @test all(isapprox.(V, optimal_V; atol = 1e-2))
    end

    @testset "MK: policy evaluation" begin
        P = 0.25 * ones(size(next_states)...)
        simulator = MDP.create_simulator(fmdp, P, 100)
        V, Q = MDP.allocate_V_and_Q(fmdp)
        MDP.mk_evaluate_policy!(Q, simulator, 1.0; maxiter = 10000)

        MDP.V_from_Q!(V, Q, P)
        expected_V = [-14.0, -20.0, -22.0, -14.0, -18.0, -20.0, -20.0, -20.0, -20.0, -18.0, -14.0, -22.0, -20.0, -14.0, 0.0]
        e = (abs.(V - expected_V))[1:end-1]
        @test max(e...) < 2
    end

    @testset "MK: ε-greedy simulation" begin
        optimal_V = [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -1.0, -3.0, -2.0, -1.0, 0.0]
        V, Q = MDP.allocate_V_and_Q(fmdp)
        MDP.Q_from_V!(Q, optimal_V, fmdp, 1.0)
        𝐩 = rand(1:4, size(next_states, 1))
        MDP.P_from_Q!(𝐩, Q)
        simulator = MDP.create_simulator(fmdp, 𝐩, 0.05, 100)
        episode = simulator(5, 2)
    end

    @testset "MK: policy optimization" begin
        𝐩 = rand(1:4, size(next_states, 1))
        simulator = MDP.create_simulator(fmdp, 𝐩, 0.05, 100)
        optimal_V = [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0, -2.0, -2.0, -3.0, -2.0, -1.0, -3.0, -2.0, -1.0, 0.0]
        V, Q = MDP.allocate_V_and_Q(fmdp)
        converged = false
        for i = 1:10
            MDP.mk_evaluate_policy!(Q, simulator, 1.0; maxiter = 10000)
            modified = MDP.P_from_Q!(𝐩, Q)
        end # for: iterations

        e = (abs.(V - optimal_V))[1:end-1]
        @test max(e...) < 4
        @info max(e...)
    end

end