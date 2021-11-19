module FiniteMarkovDecisionProcesses

abstract type FiniteMDP end

struct DeterministicFiniteMDP <: FiniteMDP
    next_state::Function
    reward::Function
    actions_no::Int
    states_no::Int
end # struct

struct StochasticFiniteMDP{T<:Real} <: FiniteMDP
    probabilities::Function
    states_no::Int
    actions_no::Int
    rewards::Array{T}
end # struct


states_no(fmdp::DeterministicFiniteMDP) = fmdp.states_no
states_no(fmdp::StochasticFiniteMDP) = fmdp.states_no

actions_no(fmdp::DeterministicFiniteMDP) = fmdp.actions_no
actions_no(fmdp::StochasticFiniteMDP) = fmdp.actions_no


include("_0_utils.jl")
include("_1_dynamic_programming.jl")

end # module