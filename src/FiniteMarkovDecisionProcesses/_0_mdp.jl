abstract type FiniteMDP end

struct DeterministicFiniteMDP <: FiniteMDP
    next_state::Function
    reward::Function
    actions_no::Int
    states_no::Int
    terminal_state::Int
end # struct

struct StochasticFiniteMDP{T<:Real} <: FiniteMDP
    probabilities::Function
    states_no::Int
    actions_no::Int
    rewards::Array{T}
    terminal_state::Int
end # struct


states_no(fmdp::DeterministicFiniteMDP) = fmdp.states_no
states_no(fmdp::StochasticFiniteMDP) = fmdp.states_no


actions_no(fmdp::DeterministicFiniteMDP) = fmdp.actions_no
actions_no(fmdp::StochasticFiniteMDP) = fmdp.actions_no


terminal_state(fmdp::DeterministicFiniteMDP) = fmdp.terminal_state
terminal_state(fmdp::StochasticFiniteMDP) = fmdp.terminal_state


act_once(fmdp::DeterministicFiniteMDP, s::Integer, a::Integer) = (fmdp.next_state(s, a), fmdp.reward(s, a))


draw_action(s::Integer, 𝐩::AbstractVector{<:Integer}) = 𝐩[s]
draw_action(s::Integer, 𝐏::AbstractMatrix{<:Real}) = (1:size(𝐏, 2))[findfirst(cumsum(𝐏[s, :]) .> rand())]

function draw_action(s::Integer, P::Union{AbstractVector{<:Integer},AbstractMatrix{<:Real}}, ε::Real)
    if rand() > ε
        a = draw_action(s, P)
    else
        a = rand(1:actions_no(fmdp))
    end
    return a
end


function create_simulator(fmdp::FiniteMDP, 𝐏::AbstractMatrix{<:Real}, maxiter::Integer)
    function inner(s0::Integer, a0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s, r = act_once(fmdp, s0, a0)
        push!(episode, (s0, a0, r))
        for _ = 1:maxiter
            if s == terminal_state(fmdp)
                # -1 is the code used to identify end of episode
                # due to achieving terminal state.
                # The remaining entries are irrelevant
                push!(episode, (-1, 1, 0.0))
                break
            end
            a = draw_action(s, 𝐏)
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        return episode
    end
end

function create_simulator(fmdp::FiniteMDP, 𝐩::AbstractVector{<:Integer}, ε::Real, maxiter::Integer)
    function inner(s0::Integer, a0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s, r = act_once(fmdp, s0, a0)
        push!(episode, (s0, a0, r))
        for _ = 1:maxiter
            if s == terminal_state(fmdp)
                # -1 is the code used to identify end of episode
                # due to achieving terminal state.
                # The remaining entries are irrelevant.
                push!(episode, (-1, 1, 0.0))
                break
            end
            if rand() > ε
                a = draw_action(s, 𝐩)
            else
                a = rand(1:actions_no(fmdp))
            end
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        return episode
    end
end
