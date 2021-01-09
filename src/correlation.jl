using Distributions
using Statistics
using StatsBase
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
5×1 Array{Float64,2}:
 0.02253061224489797
 0.0035510204081632673
 0.016000000000000004
 0.12110204081632653
 0.05355102040816326
```
"""
function bsmahal(a::Array{T,2},
                 b::Array{T,2};
                 n_boot::Int64=200)::Array{Float64,2} where T <: Number
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

"""
    shepherd(x, y[, n_boot])

Shepherd's Pi correlation, equivalent to Spearman's rho after outliers
removal.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. x and y must be independent.
- `n_boot::Int64`: Number of bootstrap samples to calculate.

Returns
-------
- r::Float64: Pi correlation coefficient,
- pval::Float64: Two-tailed adjusted p-value,
- outliers::Array{bool}: Indicate if value is an outlier or not.

**WARNING: I need to add pvalue to return statement (waiting for HypothesisTests.jl commit)**


Notes
-----
It first bootstraps the Mahalanobis distances, removes all observations
with m >= 6 and finally calculates the correlation of the remaining data.

Pi is Spearman's Rho after outlier removal.

Examples
--------

```julia-repl
julia> x = [0, 1, 2, 1, 0, 122, 1, 3, 5]
julia> y = [1, 1, 3, 0, 0, 5, 1, 3, 4]
julia> r, outliers = shepherd(x, y, n_boot=2000)
julia> r
0.834411830506179
julia> outliers'
1×9 Adjoint{Bool,BitArray{2}}:
 0  0  0  0  0  1  0  0  0
```
"""
function shepherd(x::Array{<:Number},
                  y::Array{<:Number};
                  n_boot::Int64=200)::Tuple{Float64,BitArray{2}}
    X = hcat(x, y)
    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X, n_boot=n_boot)
    # Determine outliers
    outliers = (m .>= 6)
    # Compute correlation
    # todo: add pvalue to return statement (waiting for HypothesisTests.jl commit)
    r = corspearman(x[findall(@. !outliers)], y[findall(@. !outliers)])

    return r, outliers
end


"""
    percbend(x, y[, β])

Percentage bend correlation (Wilcox 1994).

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. x and y must be independent.
- `β::Float64`: Bending constant for omega (0.0 <= β <= 0.5).

Returns
-------
- `r::Float64`: Percentage bend correlation coefficient.
- `pval::Float64`: Two-tailed p-value.

Notes
-----
Code inspired by Matlab code from Cyril Pernet and Guillaume Rousselet.

References
----------
[1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

[2] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
False Positive and Power Validation Using a New Open Source Matlab
Toolbox. Frontiers in Psychology. 2012;3:606.
doi:10.3389/fpsyg.2012.00606.

Examples
--------

```julia-repl
julia> x = [5,7,8,4,5,3,6,9]
julia> y = [8,7,4,5,89,4,1,1]
julia> r, pval = Pingouin.percbend(x, y)
julia> pval
0.39938135704241806
```
"""
function percbend(x::Array{<:Number},
                  y::Array{<:Number};
                  β::Float64=.2)::Tuple{Float64,Float64}
    X = hcat(x, y)
    nx = size(X)[1]
    M = repeat(median(X, dims=1), nx)
    W = sort(abs.(X .- M), dims=1)
    m = Int(floor((1 - β) * nx))
    ω = W[m, :]
    P = (X .- M) ./ ω'
    replace!(P, Inf => 0.0)
    replace!(P, NaN => 0.0)

    # Loop over columns
    a = zeros((2, nx))
    for c in [1, 2]
        ψ = P[:, c]
        i1 = count(ψ .< -1)
        i2 = count(ψ .> 1)
        s = X[:, c]
        s[ψ .< -1] .= 0
        s[ψ .> 1] .= 0
        pbos = (sum(s) + ω[c] * (i2 - i1)) / (length(s) - i1 - i2)
        a[c, :] = (X[:, c] .- pbos) ./ ω[c]
    end

    # Bend
    a[a .<= -1.] .= -1.
    a[a .>= 1.] .= 1.
    
    # Get r, tval, and pval
    b = a[2, :]
    a = a[1, :]
    r = sum(a .* b) / sqrt(sum(a.^2) * sum(b.^2))
    tval = r * sqrt((nx - 2) / (1 - r^2))
    pval = 2 * ccdf(TDist(nx - 2), abs(tval))

    return r, pval
end