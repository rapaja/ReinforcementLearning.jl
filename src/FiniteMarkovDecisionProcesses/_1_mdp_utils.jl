"""
    allocate_V([T=Float64], mdp, [initializer=i->rand()])

Create uninitialized matrix of state values `V`, 
given a finite Markov decission problem `fmdp`, and an optional type `T`.
Elements of the created matrices are of type `T`.

The initializer function will be applied to each element of the allocated
array in order to initialize it. The `initializer` is a function which takes
element index, and computes value for the corresponding element. If not given, 
the default `initializer` will be used, which assigns an uniform random number
between 0 and 1 to each element. 
"""
function allocate_V(T::Type, fmdp::FiniteMDP, initializer::Function)
    V = Vector{T}(undef, states_no(fmdp))
    for i in eachindex(V)
        V[i] = initializer(i)
    end
    return V
end

allocate_V(fmdp::FiniteMDP, initializer::Function) = allocate_V(Float64, fmdp, initializer)
allocate_V(T::Type, fmdp::FiniteMDP) = allocate_V(T, fmdp, i -> rand())
allocate_V(fmdp::FiniteMDP) = allocate_V(Float64, fmdp)

"""
    allocate_Q([T=Float64], mdp, [initializer=i->rand])

Create uninitialized matrix of state-action values `Q`, 
given a finite Markov decission problem `fmdp`, and an optional type `T`.
Elements of the created matrices are of type `T`. 

The initializer function will be applied to each element of the allocated
matrix in order to initialize it. The `initializer` is a function which takes
element indices, and computes value for the corresponding element. If not given, 
the default `initializer` will be used, which assigns an uniform random number
between 0 and 1 to each element. 
"""
function allocate_Q(T::Type, fmdp::FiniteMDP, initializer::Function)
    Q = Matrix{T}(undef, states_no(fmdp), actions_no(fmdp))
    for i in eachindex(Q)
        Q[i] = initializer(i)
    end
    return Q
end

allocate_Q(fmdp::FiniteMDP, initializer::Function) = allocate_Q(Float64, fmdp, initializer)
allocate_Q(T::Type, fmdp::FiniteMDP) = allocate_Q(T, fmdp, i -> rand())
allocate_Q(fmdp::FiniteMDP) = allocate_Q(Float64, fmdp)


"""
    expected_return(fmdp, s, action_or_policy, V, γ)

Expected return achieved for the finite Markov Decision Process `fmdp` (deterministic or 
stochastic) in the state `s` after taking a given action (or following the given policy, 
either deterministic or stochastic), assuming discount factor `γ`. `V` is the vector of
state values.
"""
function expected_return(fmdp::StochasticFiniteMDP, s::Integer, a::Integer, V::AbstractVector{<:Real}, γ::Real)
    ℙsr = fmdp.probabilities(s, a)
    ret = 0
    for s_next = 1:states_no(fmdp)
        for (rndx, r) in enumerate(fmdp.rewards)
            ret += ℙsr[s_next, rndx] * (r + γ * V[s_next])
        end # for: rewards
    end # for: next states
    return ret
end

expected_return(fmdp::DeterministicFiniteMDP, s::Integer, a::Integer, V::AbstractVector{<:Real}, γ::Real) =
    fmdp.reward(s, a) + γ * V[fmdp.next_state(s, a)]

function expected_return(fmdp::FiniteMDP, s::Integer, 𝐏::AbstractMatrix{<:Real}, V::AbstractVector{<:Real}, γ::Real)
    ret = 0
    for a = 1:actions_no(fmdp)
        ret += 𝐏[s, a] * expected_return(fmdp, s, a, V, γ)
    end # for: actions
    return ret
end

expected_return(fmdp::FiniteMDP, s::Integer, 𝐩::AbstractVector{<:Real}, V::AbstractVector{<:Real}, γ::Real) =
    expected_return(fmdp, s, 𝐩[s], V, γ)

"""
    Q_from_V!(Q, V, fmdp, γ)

Reevaluate (in-place) matrix of state-action values `Q`, 
given a vector of state values `V`, a finite Markov Decision Process `fmdp`, 
assuming discount factor `γ`.

`fmdp` may be either stohastic or deterministic Markov Decision Process.

Return the maximal absolute update of any element of `Q`.
"""
function Q_from_V!(Q::AbstractMatrix{<:Real}, V::AbstractVector{<:Real}, fmdp::StochasticFiniteMDP{<:Real}, γ::Real)
    Δ = -Inf
    for s = 1:states_no(fmdp)
        for a = 1:actions_no(fmdp)
            ℙsr = fmdp.probabilities(s, a)
            prev = Q[s, a]
            Q[s, a] = 0
            for s_next = 1:states_no(fmdp)
                for (rndx, r) in enumerate(fmdp.rewards)
                    Q[s, a] += ℙsr[s_next, rndx] * (r + γ * V[s_next])
                end # for: rewards
            end # for: next states
            Δ = max(Δ, abs(Q[s, a] - prev))
        end # for: actions
    end # for: states
    return Δ
end

function Q_from_V!(Q::AbstractMatrix{<:Real}, V::AbstractVector{<:Real}, mdp::DeterministicFiniteMDP, γ::Real)
    Δ = -Inf
    for s = 1:states_no(mdp)
        for a = 1:actions_no(mdp)
            prev = Q[s, a]
            Q[s, a] = mdp.reward(s, a) + γ * V[mdp.next_state(s, a)]
            Δ = max(Δ, abs(Q[s, a] - prev))
        end # for: actions
    end # for: states
    return Δ
end

"""
    V_from_Q!(V, Q, [policy])

Reevaluate (in-place) vector of state values `V`, given a vector of state-action values `Q`
and a decision policy `policy`.

If `policy` is a matrix, it will be interpreted as a stochastic decision policy,
with entry at coordinate `policy[s,a]` to be interpreted as the probability of choosing
action `a` at state `s`. If `policy` is a vector, it will be interpreted as a deterministic 
decision policy, with entry at coordinate `policy[s]` to be interpreted as the action to choose 
at state `s`.

If `policy` is not given, then the optimal decission policy according to `Q` will be used.

Return the maximal absolute update of any element of `V`.
"""
function V_from_Q!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real}, 𝐏::AbstractMatrix{<:Real})
    Δ = -Inf
    for s = 1:size(Q, 1)
        prev = V[s]
        V[s] = 0
        for a = 1:size(Q, 2)
            V[s] += 𝐏[s, a] * Q[s, a]
        end # for: actions
        Δ = max(Δ, abs(V[s] - prev))
    end # for: states
    return Δ
end

function V_from_Q!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real}, 𝐩::AbstractVector{<:Integer})
    Δ = -Inf
    for s = 1:size(Q, 1)
        prev = V[s]
        V[s] = Q[s, 𝐩[s]]
        Δ = max(Δ, abs(V[s] - prev))
    end # for: states
    return Δ
end

function V_from_Q!(V::AbstractVector{<:Real}, Q::AbstractMatrix{<:Real})
    Δ = -Inf
    for s = 1:size(Q, 1)
        prev = V[s]
        V[s] = Q[s, argmax(replace(Q[s, :], NaN => -Inf))]
        Δ = max(Δ, abs(V[s] - prev))
    end # for: states
    return Δ
end

"""
    allocate_𝐩([T=Int], Q)

Create a vector representing deterministic decission policy based on the size
of the given matrix of state-action values `Q`, and an optinal integer type `T`.

All elemenents of the resulting vector are of type `T`.

The content of the created vector is uninitialized! After creation, the content
is completely unrelated to the content of `Q`.
"""
allocate_𝐩(T::Type, Q::AbstractMatrix{<:Real}) = Vector{T}(undef, size(Q, 1))
allocate_𝐩(Q::AbstractMatrix{<:Real}) = allocate_𝐩(Int, Q)

"""
    𝐩_from_Q!(𝐩, Q)

Reevaluate (in-place) deterministic greedy policy (represented by a vector whose entry 
at index `s` corresponds to the action to take in `s`) from the given state-action matrix `Q`.

Returns boolean indicator showing whether policy has changed or not.
"""
function 𝐩_from_Q!(𝐩::AbstractVector{<:Integer}, Q::AbstractMatrix{<:Real})
    modified = false
    for s = 1:size(Q, 1)
        temp = 𝐩[s]
        𝐩[s] = argmax(replace(Q[s, :], NaN => -Inf))
        if isnan(𝐩[s])
            𝐩[s] = rand(1:size(Q, 2))
        end
        if temp != 𝐩[s]
            modified = true
        end
    end # for: states
    return modified
end
