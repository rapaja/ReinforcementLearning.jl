module FiniteMarkovDecisionProcesses

struct FiniteMarkovDecissionProblem{T<:Real}
    probabilities::Function
    states_no::Int
    actions_no::Int
    rewards::Array{T}
end # struct

include("Tabular/Tabular.jl")

end # module