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


function create_simulator(fmdp::FiniteMDP, 𝐏::AbstractMatrix{<:Real})
    function inner(s0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s = s0
        while s != terminal_state(fmdp)
            a = draw_action(s, 𝐏)
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        push!(episode, (s, 1, 0.0))
        return episode
    end
end

function create_simulator(fmdp::FiniteMDP, 𝐩::AbstractVector{<:Integer}, ε::Real)
    function inner(s0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s = s0
        while s != terminal_state(fmdp)
            if rand() > ε
                a = draw_action(s, 𝐩)
            else
                a = rand(1:actions_no(fmdp))
            end
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        push!(episode, (s, 1, 0.0))
        return episode
    end
end
