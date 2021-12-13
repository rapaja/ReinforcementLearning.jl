draw_action_from_policy(s::Integer, 𝐩::AbstractVector{<:Integer}) = 𝐩[s]
draw_action_from_policy(s::Integer, 𝐏::AbstractMatrix{<:Real}) = (1:size(𝐏, 2))[findfirst(cumsum(𝐏[s, :]) .> rand())]

function draw_action_from_policy(fmdp::FiniteMDP, s::Integer, P::Union{AbstractVector{<:Integer},AbstractMatrix{<:Real}}, ε::Real)
    if rand() > ε
        a = draw_action_from_policy(s, P)
    else
        a = random_action(fmdp)
    end
    return a
end

# TODO: This particular piece of code appears elsewhere... (v_from_Q?) merge
draw_action_from_Q(s::Integer, Q::AbstractMatrix{<:Real}) = (1:size(Q, 2))[argmax(replace(Q[s, :], NaN => -Inf))]

function draw_action_from_Q(s::Integer, Q::AbstractMatrix{<:Real}, ε::Real)
    if rand() > ε
        a = draw_action_from_Q(s, Q)
    else
        a = rand(1:size(Q, 2))
    end
    return a
end


const TERMINAL_STATE_CODE = -1
const TERMINAL_STATE_TRIPPLE = (TERMINAL_STATE_CODE, 0, 0.0)


function create_simulator_from_policy(fmdp::FiniteMDP, policy::Union{AbstractVector{<:Integer},AbstractMatrix{<:Real}}, ε, maxiter::Integer)
    function inner(s0::Integer, a0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s, r = act_once(fmdp, s0, a0)
        push!(episode, (s0, a0, r))
        for _ = 1:maxiter
            if s == terminal_state(fmdp)
                push!(episode, TERMINAL_STATE_TRIPPLE)
                break
            end
            a = draw_action_from_policy(fmdp, s, policy, ε)
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        return episode
    end
end

create_simulator_from_policy(fmdp::FiniteMDP, policy::Union{AbstractVector{<:Integer},AbstractMatrix{<:Real}}, maxiter::Integer) =
    create_simulator_from_policy(fmdp, policy, 0.0, maxiter)

function create_simulator_from_Q(fmdp::FiniteMDP, Q::AbstractMatrix{<:Real}, ε::Real, maxiter::Integer)
    function inner(s0::Integer, a0::Integer)
        episode = Vector{Tuple{Int,Int,Float64}}()
        s, r = act_once(fmdp, s0, a0)
        push!(episode, (s0, a0, r))
        for _ = 1:maxiter
            if s == terminal_state(fmdp)
                push!(episode, TERMINAL_STATE_TRIPPLE)
                break
            end
            a = draw_action_from_Q(s, Q, ε)
            (s_next, r) = act_once(fmdp, s, a)
            push!(episode, (s, a, r))
            s = s_next
        end
        return episode
    end
end

create_simulator_from_Q(fmdp::FiniteMDP, Q::AbstractMatrix{<:Real}, maxiter::Integer) =
    create_simulator_from_Q(fmdp, Q, ε, maxiter)