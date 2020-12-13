using Distributions
using Statistics
using LinearAlgebra


"""
    bsmahal(a, b[, n_boot])

Bootstraps Mahalanobis distances for Shepherd's pi correlation.

Arguments
----------
- `a::Array{<:Number,2}`: Data.
- `b::Array{<:Number,2}`: Data.
- `n_boot::Int64`: Number of bootstrap samples to calculate.

Returns
-------
- `m::Array{Float64}`: Mahalanobis distance for each row in a, averaged across all the bootstrap resamples.

Example
_______

```julia-repl
julia> a = [[1,4,5,2,5] [2,5,6,8,4]]
julia> b = [[2,5,4,8,5] [2,6,7,9,5]]
julia> bsmahal(a, b)

```
"""
function bsmahal(a::Array{T,2},
                 b::Array{T,2};
                 n_boot::Int64=200) where T <: Number
    n, m = size(b)
    MD = zeros(n, n_boot)
    xB = sample(1:n, (n_boot, n))

    # Bootstrap the MD
    for i in 1:n_boot:1
        s1 = b[xB[i, :], 1]
        s2 = b[xB[i, :], 2]
        X = hcat(s1, s2)
        mu = mean(X, dims=1)
        R = qr(X .- mu).R
        sol = R' \ (a .- mu)'
        MD[:, i] = sum(sol.^2, dims=1) .* (n - 1)
    end

    return mean(MD, dims=2)
end