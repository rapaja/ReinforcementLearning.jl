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


random_action(fmdp::FiniteMDP) = rand(1:actions_no(fmdp))
