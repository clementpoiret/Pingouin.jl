include("bayesian.jl")
include("effsize.jl")
include("power.jl")
include("utils.jl")

using DataFrames
using Distances
using Distributions
using Statistics
using StatsBase
using LinearAlgebra
using LinRegOutliers

"""
    _correl_pvalue(r, n[, k, tail])

Compute the p-value of a correlation coefficient.

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_exact_distribution
See also scipy.stats._ttest_finish

Arguments
---------
- `r::Float64`: Correlation coefficient.
- `n::Int64`: Sample size.
- `k::Int64`: Number of covariates for (semi)-partial correlation.
- `tail::Symbol`: Tail of the test.

Returns
-------
- `pval::Float64`: p-value.

Notes
-----
This uses the same approach as `scipy.stats.pearsonr` to calculate
the p-value (i.e. using a beta distribution)
"""
function _correl_pvalue(r::Float64,
    n::Int64,
    k::Int64 = 0;
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."

    # Method using a student T distribution
    ddof = n - k - 2
    tval = r * sqrt(ddof / (1 - r^2))


    if tail == :left
        pval = cdf(TDist(ddof), tval)
    elseif tail == :right
        pval = ccdf(TDist(ddof), tval)
    elseif tail == :both
        pval = 2 * ccdf(TDist(ddof), abs(tval))
    end

    return pval
end

"""
    skipped(x, y[, corr_type])

Skipped correlation (Rousselet and Pernet 2012).

Arguments
---------
- `x, y::Vector{<:Number}`: First and second set of observations. x and y must be independent.
- `corr_type::String`: Method used to compute the correlation after outlier removal. Can be either 'spearman' (default) or 'pearson'.

Returns
-------
- `r::Float64`: Skipped correlation coefficient.
- `pval::Float64`: Two-tailed p-value.
- `outliers::BitArray`: Indicate if value is an outlier or not.

Notes
-----
The skipped correlation involves multivariate outlier detection using a
projection technique (Wilcox, 2004, 2005). First, a robust estimator of
multivariate location and scatter, for instance the minimum covariance
determinant estimator (MCD; Rousseeuw, 1984; Rousseeuw and van Driessen,
1999; Hubert et al., 2008) is computed. Second, data points are
orthogonally projected on lines joining each of the data point to the
location estimator. Third, outliers are detected using a robust technique.
Finally, Spearman correlations are computed on the remaining data points
and calculations are adjusted by taking into account the dependency among
the remaining data points.

Code inspired by Matlab code from Cyril Pernet and Guillaume
Rousselet [1].

References
----------
[1] Pernet CR, Wilcox R, Rousselet GA. Robust Correlation Analyses:
    False Positive and Power Validation Using a New Open Source Matlab
    Toolbox. Frontiers in Psychology. 2012;3:606.
    doi:10.3389/fpsyg.2012.00606.
"""
function skipped(x::Vector{<:Number},
    y::Vector{<:Number};
    corr_type::String = "spearman")::Tuple{Float64,Float64,BitVector}

    @assert corr_type in ["spearman", "pearson"], "Correlation type must be one of 'spearman' (default) or 'pearson'."

    X = hcat(x, y)
    X_df = DataFrame(X, :auto)
    nrows = size(X)[1]

    mincovdet = mcd(X_df, alpha = 2.0)
    outliers_idx = mincovdet["outliers"]
    idx = BitVector(ones(nrows))
    idx[outliers_idx] .= 0

    if corr_type == "spearman"
        r = corspearman(x[idx], y[idx])
    elseif corr_type == "pearson"
        r = cor(x[idx], y[idx])
    end

    @warn "Warning: p-value maybe different than expected."
    pval = _correl_pvalue(r, nrows, 0, tail = :both)

    return r, pval, @. !idx
end

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
    n_boot::Int64 = 200)::Array{Float64,2} where {T<:Number}
    n, m = size(b)
    MD = zeros(n, n_boot)
    xB = sample(1:n, (n_boot, n))

    # Bootstrap the MD
    for i = 1:n_boot:1
        s1 = b[xB[i, :], 1]
        s2 = b[xB[i, :], 2]
        X = hcat(s1, s2)
        mu = mean(X, dims = 1)
        R = qr(X .- mu).R
        sol = R' \ (a .- mu)'
        MD[:, i] = sum(sol .^ 2, dims = 1) .* (n - 1)
    end

    return mean(MD, dims = 2)
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
julia> r, pval, outliers = shepherd(x, y, n_boot=2000);
julia> r
0.88647133421903
julia> pval
0.0033537058539450776
julia> outliers'
1×9 Adjoint{Bool,BitArray{2}}:
 0  0  0  0  0  1  0  0  0
```
"""
function shepherd(x::Array{<:Number},
    y::Array{<:Number};
    n_boot::Int64 = 200)::Tuple{Float64,Float64,BitArray{2}}
    X = hcat(x, y)
    # Bootstrapping on Mahalanobis distance
    m = bsmahal(X, X, n_boot = n_boot)
    # Determine outliers
    outliers = (m .>= 6)
    # Compute correlation
    r = corspearman(x[findall(@. !outliers)], y[findall(@. !outliers)])

    @warn "Warning: p-value maybe different than expected."
    pval = _correl_pvalue(r, size(X)[1], 0, tail = :both)

    return r, pval, outliers
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
    β::Float64 = 0.2)::Tuple{Float64,Float64}
    X = hcat(x, y)
    nx = size(X)[1]
    M = repeat(median(X, dims = 1), nx)
    W = sort(abs.(X .- M), dims = 1)
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
        s[ψ.<-1] .= 0
        s[ψ.>1] .= 0
        pbos = (sum(s) + ω[c] * (i2 - i1)) / (length(s) - i1 - i2)
        a[c, :] = (X[:, c] .- pbos) ./ ω[c]
    end

    # Bend
    a[a.<=-1.0] .= -1.0
    a[a.>=1.0] .= 1.0

    # Get r, tval, and pval
    b = a[2, :]
    a = a[1, :]
    r = sum(a .* b) / sqrt(sum(a .^ 2) * sum(b .^ 2))
    tval = r * sqrt((nx - 2) / (1 - r^2))
    pval = 2 * ccdf(TDist(nx - 2), abs(tval))

    return r, pval
end


"""
    bicor(x, y[, c])

Biweight midcorrelation.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. x and y must be independent.
- `c::Float64`: Tuning constant for the biweight estimator (default = 9.0).

Returns
-------
- `r::Float64`: Correlation coefficient.
- `pval::Float64`: Two-tailed p-value.

Notes
-----
This function will return (NaN, NaN) if mad(x) == 0 or mad(y) == 0.

References
----------
https://en.wikipedia.org/wiki/Biweight_midcorrelation

https://docs.astropy.org/en/stable/api/astropy.stats.biweight.biweight_midcovariance.html

Langfelder, P., & Horvath, S. (2012). Fast R Functions for Robust
Correlations and Hierarchical Clustering. Journal of Statistical Software,
46(11). https://www.ncbi.nlm.nih.gov/pubmed/23050260

Examples
--------

```julia-repl
julia> x = [5,7,8,4,5,3,6,9]
julia> y = [8,7,4,5,89,4,1,1]
julia> r, pval = Pingouin.bicor(x, y)
julia> pval
0.4157350278895959
```
"""
function bicor(x::Array{<:Number},
    y::Array{<:Number};
    c::Float64 = 9.0)
    # Calculate median
    nx = size(x)[1]
    x_median = median(x)
    y_median = median(y)

    # Raw median absolute deviation
    x_mad = median(abs.(x .- x_median))
    y_mad = median(abs.(y .- y_median))
    if x_mad == 0 || y_mad == 0
        # From Langfelder and Horvath 2012:
        # "Strictly speaking, a call to bicor in R should return a missing
        # value if mad(x) = 0 or mad(y) = 0." This avoids division by zero.
        return NaN, NaN
    end

    # Calculate weights
    u = (x .- x_median) ./ (c * x_mad)
    v = (y .- y_median) ./ (c * y_mad)
    w_x = (1 .- u .^ 2) .^ 2 .* ((1 .- abs.(u)) .> 0)
    w_y = (1 .- v .^ 2) .^ 2 .* ((1 .- abs.(v)) .> 0)

    # Normalize x and y by weights
    x_norm = (x .- x_median) .* w_x
    y_norm = (y .- y_median) .* w_y
    denom = (sqrt(sum(x_norm .^ 2)) * sqrt(sum(y_norm .^ 2)))

    # Correlation coefficient
    r = sum(x_norm .* y_norm) / denom
    pval = _correl_pvalue(r, nx, 0, tail = :both)

    return r, pval
end


"""
    corr(x, y[, tail, method, kwargs...])

(Robust) correlation between two variables.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. ``x`` and ``y`` must be independent.
- `tail::Symbol`: Defines the alternative hypothesis, or tail of the correlation. Must be one of :both (default), :right or :left. Both :right and :left return a one-sided p-value. :right tests against the alternative hypothesis that the correlation is positive (greater than zero), :left tests against the hypothesis that the correlation is negative.
- `method::String`: Correlation type:
    * `pearson`: Pearson \$r\$ product-moment correlation
    * `spearman`: Spearman's \$\rho\$ rank-order correlation
    * `kendall`: Kendall's \$\tau\$ correlation (for ordinal data)
    * `bicor`: Biweight midcorrelation (robust)
    * `percbend`: Percentage bend correlation (robust)
    * `shepherd`: Shepherd's pi correlation (robust)
    * `skipped`: Skipped correlation (robust)
- `kwargs` (optional): Additional keyword arguments passed to the correlation function.

Returns
-------
- `stats::DataFrame`:
    * `n`: Sample size (after removal of missing values)
    * `outliers`: number of outliers, only if a robust method was used
    * `r`: Correlation coefficient
    * `CI95`: 95% parametric confidence intervals around `r`
    * `r2`: R-squared (`= r^2`)
    * `adj_r2`: Adjusted R-squared
    * `p-val`: tail of the test
    * `BF10`: Bayes Factor of the alternative hypothesis (only for Pearson correlation)
    * `power`: achieved power of the test (= 1 - type II error).

See also
--------
[`pairwise_corr`](@ref): Pairwise correlation between columns of a pandas DataFrame
[`partial_corr`](@ref): Partial correlation
[`rm_corr`](@ref): Repeated measures correlation

Notes
-----
The `Pearson correlation coefficient
<https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
measures the linear relationship between two datasets. Strictly speaking,
Pearson's correlation requires that each dataset be normally distributed.
Correlations of -1 or +1 imply a perfect negative and positive linear
relationship, respectively, with 0 indicating the absence of association.

\$\$ r_{xy} = \\frac{\\sum_i(x_i - \\bar{x})(y_i - \\bar{y})} {\\sqrt{\\sum_i(x_i - \\bar{x})^2} \\sqrt{\\sum_i(y_i - \\bar{y})^2}} = \\frac{\\text{cov}(x, y)}{\\sigma_x \\sigma_y} \$\$

where \$cov\$ is the sample covariance and \$sigma\$
is the sample standard deviation.

If `method='pearson'`, The Bayes Factor is calculated using the
`Pingouin.bayesfactor_pearson` function.

The `Spearman correlation coefficient
<https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
is a non-parametric measure of the monotonicity of the relationship between
two datasets. Unlike the Pearson correlation, the Spearman correlation does
not assume that both datasets are normally distributed. Correlations of -1
or +1 imply an exact negative and positive monotonic relationship,
respectively. Mathematically, the Spearman correlation coefficient is
defined as the Pearson correlation coefficient between the
`rank variables <https://en.wikipedia.org/wiki/Ranking>`.

The `Kendall correlation coefficient
<https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
is a measure of the correspondence between two rankings. Values also range
from -1 (perfect disagreement) to 1 (perfect agreement), with 0 indicating
the absence of association. Consistent with
:py:func:`scipy.stats.kendalltau`, Pingouin returns the Tau-b coefficient,
which adjusts for ties:

\$\$ \\tau_B = \\frac{(P - Q)}{\\sqrt{(P + Q + T) (P + Q + U)}} \$\$

where \$P\$ is the number of concordant pairs, \$Q\$ the number of
discordand pairs, \$T\$ the number of ties in x, and \$U\$
the number of ties in y.

The `biweight midcorrelation
<https://en.wikipedia.org/wiki/Biweight_midcorrelation>` and
percentage bend correlation [1]_ are both robust methods that
protects against *univariate* outliers by down-weighting observations that
deviate too much from the median.

The Shepherd pi [2] correlation and skipped [3], [4] correlation are
both robust methods that returns the Spearman correlation coefficient after
removing *bivariate* outliers. Briefly, the Shepherd pi uses a
bootstrapping of the Mahalanobis distance to identify outliers, while the
skipped correlation is based on the minimum covariance determinant
(which requires scikit-learn). Note that these two methods are
significantly slower than the previous ones.

The confidence intervals for the correlation coefficient are estimated
using the Fisher transformation.

**important: Please note that rows with missing values (NaN) will be automatically removed in a later version. For now, please remove them before calling the function.**

References
----------
[1] Wilcox, R.R., 1994. The percentage bend correlation coefficient.
Psychometrika 59, 601–616. https://doi.org/10.1007/BF02294395

[2] Schwarzkopf, D.S., De Haas, B., Rees, G., 2012. Better ways to
improve standards in brain-behavior correlation analysis. Front.
Hum. Neurosci. 6, 200. https://doi.org/10.3389/fnhum.2012.00200

[3] Rousselet, G.A., Pernet, C.R., 2012. Improving standards in
brain-behavior correlation analyses. Front. Hum. Neurosci. 6, 119.
https://doi.org/10.3389/fnhum.2012.00119

[4] Pernet, C.R., Wilcox, R., Rousselet, G.A., 2012. Robust correlation
analyses: false positive and power validation using a new open
source matlab toolbox. Front. Psychol. 3, 606.
https://doi.org/10.3389/fpsyg.2012.00606

Examples
--------
1. Pearson correlation

```julia-repl
julia> x = [1,4,2,5,8,9,6,5,4]
julia> y = [4,5,8,7,5,4,1,2,5]
julia> # Compute Pearson correlation
julia> Pingouin.corr(x, y)
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[59]:11
1×9 DataFrame
 Row │ n      outliers  r          CI95                  r2        adj_r2     p_val     BF10      power    
     │ Int64  Int64     Float64    Array…                Float64   Float64    Float64   Float64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  -0.299022  [-0.803566, 0.45557]  0.089414  -0.214115  0.434421  0.533201  0.124076
```

2. Pearson correlation with two outliers

```julia-repl
julia> x[3], y[5] = 500, -470
julia> Pingouin.corr(x, y)
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[59]:11
1×9 DataFrame
 Row │ n      outliers  r         CI95                   r2         adj_r2     p_val     BF10      power     
     │ Int64  Int64     Float64   Array…                 Float64    Float64    Float64   Float64   Float64   
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  0.127003  [-0.586595, 0.729587]  0.0161297  -0.311827  0.744724  0.426023  0.0614084
```

3. Spearman correlation (robust to outliers)

```julia-repl
julia> Pingouin.corr(x, y, method="spearman")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[68]:15
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2     p_val     BF10     power     
     │ Int64  Int64     Float64     Array…                 Float64     Float64    Float64   Float64  Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  -0.0423729  [-0.687157, 0.639752]  0.00179546  -0.330939  0.913808      NaN  0.0502253
```

4. Biweight midcorrelation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="bicor")
1×9 DataFrame
 Row │ n      outliers  r          CI95                  r2         adj_r2    p_val     BF10     power     
     │ Int64  Int64     Float64    Array…                Float64    Float64   Float64   Float64  Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  -0.170727  [-0.74983, 0.556492]  0.0291476  -0.29447  0.660536      NaN  0.0718702
```

5. Percentage bend correlation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="percbend")
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2     p_val     BF10     power     
     │ Int64  Int64     Float64     Array…                 Float64     Float64    Float64   Float64  Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  0.00520452  [-0.661203, 0.667021]  2.70871e-5  -0.333297  0.989398      NaN  0.0488702
```

6. Shepherd's pi correlation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="shepherd")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main ~/Sync/Projects/Julia/Pingouin/Pingouin.jl/src/correlation.jl:642
┌ Warning: Warning: p-value maybe different than expected.
└ @ Main ~/Sync/Projects/Julia/Pingouin/Pingouin.jl/src/correlation.jl:232
1×9 DataFrame
 Row │ n      outliers  r         CI95                   r2          adj_r2     p_val    BF10     power     
     │ Int64  Int64     Float64   Array…                 Float64     Float64    Float64  Float64  Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         1  0.097561  [-0.651931, 0.750629]  0.00951814  -0.386675  0.80282      NaN  0.0562064
```

7. One-tailed Pearson correlation

```julia-repl
julia> Pingouin.corr(x, y, tail=:right, method="pearson")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[68]:11
1×9 DataFrame
 Row │ n      outliers  r         CI95              r2         adj_r2     p_val     BF10      power     
     │ Int64  Int64     Float64   Array…            Float64    Float64    Float64   Float64   Float64   
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  0.127003  [-0.495871, 1.0]  0.0161297  -0.311827  0.372362  0.525049  0.0970358
```
"""
function corr(x::Array{<:Number},
    y::Array{<:Number};
    tail::Symbol = :both,
    method::String = "pearson",
    kwargs...)

    # Remove rows with missing values
    x, y = remove_na(x, y, paired = true)

    # todo: update when p-values available in hypothesistests.jl
    @assert tail in [:both, :left, :right] "Tail must be :both, :left or :right."

    nx = size(x)[1]
    r = pval = bf10 = pr = NaN
    outliers = 0
    # Compute correlation coefficient
    if method == "pearson"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = cor(x, y)
        pval = _correl_pvalue(r, nx, 0, tail = tail)
        bf10 = bayesfactor_pearson(r, nx, tail = tail)
    elseif method == "spearman"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = corspearman(x, y)
        pval = _correl_pvalue(r, nx, 0, tail = tail)
    elseif method == "kendall"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = corkendall(x, y)
        pval = _correl_pvalue(r, nx, 0, tail = tail)
    elseif method == "bicor"
        c = get(kwargs, :c, 9.0)
        r, pval = bicor(x, y, c = c)
    elseif method == "percbend"
        β = get(kwargs, :β, 0.2)
        r, pval = percbend(x, y, β = β)
    elseif method == "shepherd"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        n_boot = get(kwargs, :n_boot, 200)
        r, pval, outliers = shepherd(x, y, n_boot = n_boot)
        outliers = sum(outliers)
    elseif method == "skipped"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        corr_type = get(kwargs, :corr_type, "spearman")
        r, pval, outliers = skipped(x, y, corr_type = corr_type)
        outliers = sum(outliers)
    else
        throw(DomainError(method, "Method not recognized."))
    end

    if r == NaN
        return DataFrame(n = nx,
            r = NaN,
            CI95 = NaN,
            r2 = NaN,
            adj_r2 = NaN,
            p_val = NaN,
            BF10 = NaN,
            power = NaN)
    end

    n_clean = nx - outliers

    if abs(r) == 1
        ci = [r, r]
        pr = 1
    else
        # Compute the parametric 95% confidence interval and power
        ci = compute_esci(stat = r,
            nx = n_clean,
            ny = n_clean,
            eftype = "r",
            decimals = 6,
            tail = tail)
        pr = power_corr(r, nx, nothing, 0.05, tail = tail)
    end

    # Compute r2 and adj_r2
    r2 = r^2
    adj_r2 = 1 - (((1 - r2) * (n_clean - 1)) / (n_clean - 3))

    # Recompute p-value if tail is one-sided
    if tail != :both
        pval = _correl_pvalue(r, n_clean, 0, tail = tail)
    end

    return DataFrame(n = nx,
        outliers = outliers,
        r = r,
        CI95 = [ci],
        r2 = r2,
        adj_r2 = adj_r2,
        p_val = pval,
        BF10 = bf10,
        power = pr)
end


"""
    partial_corr(data, x, y, covar, x_covar, y_covar, tail, method)

Partial and semi-partial correlation.

Arguments
----------
- `data::DataFrame`: Dataframe,
- `x, y::string`: x and y. Must be names of columns in `data`,
- `covar::Union{Array{String,1},String}`: Covariate(s). Must be a names of columns in `data`. Use a list if there are two or more covariates.
- `x_covar::Union{Array{String,1},String}`: Covariate(s) for the `x` variable. This is used to compute semi-partial correlation (i.e. the effect of `x_covar` is removed from `x` but not from `y`). Note that you cannot specify both `covar` and `x_covar`.
- `y_covar::Union{Array{String,1},String}`: Covariate(s) for the `y` variable. This is used to compute semi-partial correlation (i.e. the effect of `y_covar` is removed from `y` but not from `x`). Note that you cannot specify both `covar` and `y_covar`.
- `tail::Symbol`: Specify whether to return `:onesided` or `:both` p-value. Note that the former are simply half the latter.
- `method::String`: Correlation type:
    * `"pearson"`: Pearson \$r\$ product-moment correlation
    * `"spearman"`: Spearman \$\rho\$ rank-order correlation
    * `"kendall"`: Kendall's \$\tau_B\$ correlation (for ordinal data)
    * `"bicor"`: Biweight midcorrelation (robust)
    * `"percbend"`: Percentage bend correlation (robust)
    * `"shepherd"`: Shepherd's pi correlation (robust)
    * `"skipped"`: Skipped correlation (robust)

Returns
-------
- `stats::DataFrame`
    * `"n"`: Sample size (after removal of missing values)
    * `"outliers"`: number of outliers, only if a robust method was used
    * `"r"`: Correlation coefficient
    * `"CI95"`: 95% parametric confidence intervals around ``r``
    * `"r2"`: R-squared (``= r^2``)
    * `"adj_r2"`: Adjusted R-squared
    * `"p-val"`: tail of the test
    * `"BF10"`: Bayes Factor of the alternative hypothesis (only for Pearson correlation)
    * `"power"`: achieved power of the test (= 1 - type II error).

Notes
-----
From [1]:

    *With partial correlation, we find the correlation between x
    and y holding C constant for both x and
    y. Sometimes, however, we want to hold C constant for
    just x or just y. In that case, we compute a
    semi-partial correlation. A partial correlation is computed between
    two residuals. A semi-partial correlation is computed between one
    residual and another raw (or unresidualized) variable.*

Note that if you are not interested in calculating the statistics and
p-values but only the partial correlation matrix, a (faster)
alternative is to use the `pingouin.pcorr` method (see example 4).

Rows with missing values are automatically removed from data. Results have
been tested against the
`ppcor <https://cran.r-project.org/web/packages/ppcor/index.html>`
R package.

References
----------
[1] http://faculty.cas.usf.edu/mbrannick/regression/Partial.html

Examples
--------
1. Partial correlation with one covariate

```julia-repl
julia> using Pingouin
julia> df = Pingouin.read_dataset("partial_corr")
julia> Pingouin.partial_corr(df, x="x", y="y", covar="cv1")
1×4 DataFrame
 Row │ n      r         CI95                  p_val      
     │ Int64  Float64   Array…                Float64    
─────┼───────────────────────────────────────────────────
   1 │    30  0.568169  [0.254702, 0.773586]  0.00130306
```

2. Spearman partial correlation with several covariates

```julia-repl
julia> # Partial correlation of x and y controlling for cv1, cv2 and cv3
julia> Pingouin.partial_corr(df, x="x", y="y", covar=["cv1", "cv2", "cv3"], method="spearman")
1×4 DataFrame
 Row │ n      r         CI95                  p_val      
     │ Int64  Float64   Array…                Float64    
─────┼───────────────────────────────────────────────────
   1 │    30  0.520921  [0.175685, 0.752059]  0.00533619
```

3. Semi-partial correlation on x
# todo: verify against python
```julia-repl
julia> partial_corr(df, x="x", y="y", x_covar=["cv1", "cv2", "cv3"])
1×4 DataFrame
 Row │ n      r         CI95                  p_val      
     │ Int64  Float64   Array…                Float64    
─────┼───────────────────────────────────────────────────
   1 │    30  0.492601  [0.138516, 0.735022]  0.00904416
```
"""
function partial_corr(data::DataFrame;
    x::Union{String,Symbol},
    y::Union{String,Symbol},
    covar::Union{Array{T,1},Nothing,String,Symbol} = nothing,
    x_covar::Union{Array{T,1},Nothing,String,Symbol} = nothing,
    y_covar::Union{Array{T,1},Nothing,String,Symbol} = nothing,
    tail::Symbol = :both,
    method::String = "pearson")::DataFrame where {T<:Union{Nothing,String,Symbol}}
    # todo: multiple dispatch

    @assert tail in [:both, :left, :right] "Tail must be :both, :left or :right."
    @assert method in ["pearson", "spearman"] "Method must be \"pearson\" or \"spearman\" for partial correlation."
    @assert size(data)[1] > 2 "Data must have at least 3 samples."

    if (covar !== nothing) & (x_covar !== nothing || y_covar !== nothing)
        throw(DomainError([x_covar, y_covar], "Cannot specify both covar and {x,y}_covar."))
    end

    @assert x != covar "x and covar must be independant."
    @assert y != covar "y and covar must be independant."
    @assert x != y "x and y must be independant."

    # todo: multiple dispatch
    if !isa(covar, Array)
        covar = [covar]
    end
    if !isa(x_covar, Array)
        x_covar = [x_covar]
    end
    if !isa(y_covar, Array)
        y_covar = [y_covar]
    end

    col = [i for i in [x,
        y,
        covar...,
        x_covar...,
        y_covar...] if i !== nothing]

    @assert all([Symbol(c) in propertynames(data) for c in col]) "columns are not in dataframe."

    # Remove rows with NaNs
    data = data[completecases(data), col]
    n, k = size(data)
    k -= 2
    @assert n > 2 "Data must have at least 3 non-NaN samples."

    # Calculate the partial corrrelation matrix - similar to pingouin.pcorr()
    if method == "spearman"
        # Convert the data to rank, similar to R cov()
        V = copy(data)
        for c in col
            transform!(V, c => (x -> tiedrank(x)) => c)
        end
        V = cov(Matrix(V))
    else
        V = cov(Matrix(data))
    end

    Vi = pinv(V) # Inverse covariance matrix
    Vi_diag = diag(Vi)
    D = diagm(sqrt.(1 ./ Vi_diag))
    pcor = -1 .* (D * Vi * D) # Partial correlation matrix

    if covar !== nothing
        r = pcor[1, 2]
    else
        # Semi-partial correlation matrix
        spcor = pcor ./ sqrt.(diag(V)) ./ sqrt.(abs.(Vi_diag .- Vi .^ 2 ./ Vi_diag))

        if y_covar !== nothing
            r = spcor[1, 2] # y_covar is removed from y
        else
            r = spcor[2, 1] # x_covar is removed from x
        end
    end

    if isnan(r)
        # Correlation failed. Return NaN. When would this happen?
        return DataFrame(n = n,
            r = NaN,
            CI95 = NaN,
            p_val = NaN)
    end

    # Compute the two-sided p-value and confidence intervals
    # https://online.stat.psu.edu/stat505/lesson/6/6.3
    pval = _correl_pvalue(r, n, k, tail = tail)
    ci = compute_esci(stat = r,
        nx = (n - k),
        ny = (n - k),
        eftype = "r",
        decimals = 6,
        tail = tail)

    return DataFrame(n = n,
        r = r,
        CI95 = [ci],
        p_val = pval)
end

"""
    pcorr(data)

Partial correlation matrix of a DataFrame.

Arguments
---------
- `data::DataFrame`: DataFrame.

Returns
-------
- `pcor_df::DataFrame`: Partial correlation matrix.

Notes
-----
This function calculates the pairwise partial correlations for each pair of
variables in a `DataFrame` given all the others. It has
the same behavior as the pcor function in the
https://cran.r-project.org/web/packages/ppcor/index.html R package.

Note that this function only returns the raw Pearson correlation
coefficient. If you want to calculate the test statistic and p-values, or
use more robust estimates of the correlation coefficient, please refer to
the `Pingouin.pairwise_corr` or `Pingouin.partial_corr`
functions.

Examples
--------

```julia-repl
julia> using Pingouin
julia> df = Pingouin.read_dataset("mediation");
julia> pcor_df = Pingouin.pcorr(df);
julia> pcor_df[!, 2:end] = round.(pcor_df[!, 2:end], digits=3);
julia> pcor_df
7×8 DataFrame
 Row │ name    X        M        Y        Mbin     Ybin     W1       W2      
     │ String  Float64  Float64  Float64  Float64  Float64  Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────
   1 │ X         1.0      0.359    0.074   -0.019   -0.147   -0.148   -0.067
   2 │ M         0.359    1.0      0.555   -0.024   -0.112   -0.138   -0.176
   3 │ Y         0.074    0.555    1.0     -0.001    0.169    0.101    0.108
   4 │ Mbin     -0.019   -0.024   -0.001    1.0     -0.08    -0.032   -0.04
   5 │ Ybin     -0.147   -0.112    0.169   -0.08     1.0     -0.0     -0.14
   6 │ W1       -0.148   -0.138    0.101   -0.032   -0.0      1.0     -0.394
   7 │ W2       -0.067   -0.176    0.108   -0.04    -0.14    -0.394    1.0
```

On a subset of columns

```julia-repl
julia> Pingouin.pcorr(df[!, [:X, :Y, :M]])
3×4 DataFrame
 Row │ name    X          Y          M        
     │ String  Float64    Float64    Float64  
─────┼────────────────────────────────────────
   1 │ X       1.0        0.0366487  0.412804
   2 │ Y       0.0366487  1.0        0.54014
   3 │ M       0.412804   0.54014    1.0
```
"""
function pcorr(data::DataFrame)::DataFrame
    V = cov(Matrix(data))
    Vi = pinv(V)
    D = diagm(sqrt.(1 ./ diag(Vi)))
    pcor = -1 .* (D * Vi * D)
    pcor[diagind(pcor)] .= 1
    pcor_df = DataFrame(pcor, propertynames(data))
    # todo: when DataFrame will support row names, add them
    insertcols!(pcor_df, 1, :name => names(data))

    return pcor_df
end

"""
    rm_corr(data, x, y, subject)

Repeated measures correlation.

Arguments
---------
- `data::Dataframe`.
- `x, y::Union{String,Symbol}`: Name of columns in `data` containing the two dependent variables.
- `subject::Union{String,Symbol}`: Name of column in `data` containing the subject indicator.

Returns
-------
`stats::DataFrame`

    * `"r"`: Repeated measures correlation coefficient
    * `"dof"`: Degrees of freedom
    * `"pval"`: p-value
    * `"CI95"`: 95% parametric confidence intervals
    * `"power"`: achieved power of the test (= 1 - type II error).

See also
--------
[`plot_rm_corr`](@ref)

Notes
-----
Repeated measures correlation (rmcorr) is a statistical technique for determining the common
within-individual association for paired measures assessed on two or more occasions for
multiple individuals.

From `Bakdash and Marusich (2017) https://doi.org/10.3389/fpsyg.2017.00456:

*Rmcorr accounts for non-independence among observations using analysis
of covariance (ANCOVA) to statistically adjust for inter-individual
variability. By removing measured variance between-participants,
rmcorr provides the best linear fit for each participant using parallel
regression lines (the same slope) with varying intercepts.
Like a Pearson correlation coefficient, the rmcorr coefficient
is bounded by − 1 to 1 and represents the strength of the linear
association between two variables.*

Results have been tested against the `rmcorr <https://github.com/cran/rmcorr>` R package.

Missing values are automatically removed from the dataframe (listwise deletion).

Examples
--------

```julia-repl
julia> using Pingouin
julia> df = Pingouin.read_dataset("rm_corr");
julia> Pingouin.rm_corr(df, "x", "y", "subject")
                r  dof      pval           CI95%     power
rm_corr -0.50677   38  0.000847  [-0.71, -0.23]  0.929579
```

# todo: plot
"""
function rm_corr(data::DataFrame,
    x::Union{String,Symbol},
    y::Union{String,Symbol},
    subject::Union{String,Symbol})

    error("Not implemented yet")

    @assert all([Symbol(c) in propertynames(data) for c in [x, y, subject]]) "columns are not in dataframe."
    @assert all([eltype(data[!, c]) <: Number for c in [x, y]]) "columns must be numeric."
    @assert length(unique(data[!, subject])) > 3 "subject column must contain at least 3 unique subjects."

    data = data[!, [x, y, subject]]
    # Remove missing values
    allowmissing!(data)
    foreach(c -> replace!(c, NaN => missing), eachcol(data))
    foreach(c -> replace!(c, nothing => missing), eachcol(data))
    dropmissing!(data)

    # todo: implement ancova
end

"""
    distance_corr(x, y[, tail, n_boot])

Distance correlation between two arrays.

Statistical significance (p-value) is evaluated with a permutation test.

Arguments
---------
- `x, y::Array{<:Number}`: 1D or 2D input arrays, shape (n_samples, n_features). `x` and `y` must have the same number of samples and must not contain missing values.
- `tail::Symbol`: Alternative of the test. Can be either :both, :right (default) or :left. To be consistent with the original R implementation, the default is to calculate the one-sided "greater" p-value.
- `n_boot::Union{Int,Nothing}`: Number of bootstrap to perform. If nothing, no bootstrapping is performed and the function only returns the distance correlation (no p-value). Default is 1000 (thus giving a precision of 0.001).

Returns
-------
- `dcor::Float64`: Sample distance correlation (range from 0 to 1).
- `pval::Float64`: P-value.

Notes
-----
From Wikipedia:

Distance correlation is a measure of dependence between two paired
random vectors of arbitrary, not necessarily equal, dimension. The
distance correlation coefficient is zero if and only if the random
vectors are independent. Thus, distance correlation measures both
linear and nonlinear association between two random variables or
random vectors. This is in contrast to Pearson's correlation, which can
only detect linear association between two random variables.

The distance correlation of two random variables is obtained by
dividing their distance covariance by the product of their distance
standard deviations:

``\\text{dCor}(X, Y) = \\frac{\\text{dCov}(X, Y)} {\\sqrt{\\text{dVar}(X) \\cdot \\text{dVar}(Y)}}``

where ``\\text{dCov}(X, Y)`` is the square root of the arithmetic
average of the product of the double-centered pairwise Euclidean distance
matrices.

Note that by contrast to Pearson's correlation, the distance correlation
cannot be negative, i.e ``0 \\leq \\text{dCor} \\leq 1```.

Results have been tested against the `energy https://cran.r-project.org/web/packages/energy/energy.pdf` R package.

References
----------

* https://en.wikipedia.org/wiki/Distance_correlation

* Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
    Measuring and testing dependence by correlation of distances.
    The annals of statistics, 35(6), 2769-2794.

* https://gist.github.com/satra/aa3d19a12b74e9ab7941

* https://gist.github.com/wladston/c931b1495184fbb99bec

Examples
--------
1. With two 1D vectors

```julia-repl
julia> using Pingouin
julia> a = [1, 2, 3, 4, 5]
julia> b = [1, 2, 9, 4, 4]
julia> dcor, pval = distance_corr(a, b)
(0.7626762424168667, 0.233)
```

2. With two 2D arrays and no p-value

```julia-repl
julia> using Pingouin
julia> a = rand(10, 10);
julia> b = rand(10, 10);
julia> dcor, pval = distance_corr(a, b, n_boot=nothing)
(0.9223515905902638, nothing)
```
"""
function _dcorr(y::Matrix{<:Number},
    n²::Int,
    A::Matrix{<:Number},
    dcov2_xx::Float64)

    # Pairwise Euclidean distances
    b = pairwise(Euclidean(), y, y, dims = 1)
    # Double centering
    B = b .- mean(b, dims = 1)[:, :] .- mean(b, dims = 2)[:, :] .+ mean(b)
    # Compute squared distance covariances
    dcov2_yy = dot(B, B) / n²
    dcov2_xy = dot(A, B) / n²

    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
function distance_corr(x::Vector{<:Number},
    y::Vector{<:Number};
    tail::Symbol = :right,
    n_boot::Union{Int,Nothing} = 1000)::Tuple{Float64,Union{Float64,Nothing}}

    @assert tail in [:both, :left, :right] "`tail` must be :both, :left, or :right."
    # Check for NaN values
    @assert all(@. !isnan(x)) & all(@. !isnan(y)) "`x` and `y` must not contain NaN values."
    @assert length(x) == length(y) "`x` and `y` must have the same number of samples."

    x = reshape(x, length(x), 1)
    y = reshape(y, length(y), 1)

    return distance_corr(x, y, tail = tail, n_boot = n_boot)
end
function distance_corr(x::Matrix{<:Number},
    y::Matrix{<:Number};
    tail::Symbol = :right,
    n_boot::Union{Int,Nothing} = 1000)::Tuple{Float64,Union{Float64,Nothing}}

    @assert tail in [:both, :left, :right] "`tail` must be :both, :left, or :right."
    # Check for NaN values
    @assert all(@. !isnan(x)) & all(@. !isnan(y)) "`x` and `y` must not contain NaN values."
    @assert length(x) == length(y) "`x` and `y` must have the same number of samples."

    n = size(x)[1]
    n² = n^2

    # Process first array to avoid redundancy when performing bootstrap
    a = pairwise(Euclidean(), x, x, dims = 1)
    A = a .- mean(a, dims = 1)[:, :] .- mean(a, dims = 2)[:, :] .+ mean(a)
    dcov2_xx = dot(A, A) / n²

    # Process second array and compute final distance correlation
    dcor = _dcorr(y, n², A, dcov2_xx)

    # Compute one-sided p-value using a bootstrap procedure
    n_boot = n_boot === nothing ? 0 : n_boot
    if n_boot > 1
        bootsam = sample(1:n, (n_boot, n))
        bootstat = Vector{Float64}(undef, n_boot)
        for i = 1:n_boot
            bootstat[i] = _dcorr(y[bootsam[i, :]][:, :], n², A, dcov2_xx)
        end

        pval = _perm_pval(bootstat, dcor, tail = tail)

        return dcor, pval
    else
        return dcor, nothing
    end
end
