include("bayesian.jl")
include("effsize.jl")

using DataFrames
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
               c::Float64=9.0)
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
    w_x = (1 .- u.^2).^2 .* ((1 .- abs.(u)) .> 0)
    w_y = (1 .- v.^2).^2 .* ((1 .- abs.(v)) .> 0)

    # Normalize x and y by weights
    x_norm = (x .- x_median) .* w_x
    y_norm = (y .- y_median) .* w_y
    denom = (sqrt(sum(x_norm.^2)) * sqrt(sum(y_norm.^2)))

    # Calculate r, t, and two-sided p-value
    r = sum(x_norm .* y_norm) / denom
    tval = r * sqrt((nx - 2) / (1 - r^2))
    pval = 2 * ccdf(TDist(nx - 2), abs(tval))

    return r, pval
end


"""
    corr(x, y[, tail, method])

(Robust) correlation between two variables.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. ``x`` and ``y`` must be independent.
- `tail::String`: Specify whether to return `one-sided` or `two-sided` p-value. Note that the former are simply half the latter.
- `method::String`: Correlation type:
    * `pearson`: Pearson \$r\$ product-moment correlation
    * `spearman`: Spearman's \$\rho\$ rank-order correlation
    * `kendall`: Kendall's \$\tau\$ correlation (for ordinal data)
    * `bicor`: Biweight midcorrelation (robust)
    * `percbend`: Percentage bend correlation (robust)
    * `shepherd`: Shepherd's pi correlation (robust)

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
 Row │ n      outliers  r          CI95                  r2        adj_r2     p_val    BF10      power   
     │ Int64  Float64   Float64    Array…                Float64   Float64    Float64  Float64   Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   -0.299022  [-0.803566, 0.45557]  0.089414  -0.214115     NaN   0.533201     NaN
```

2. Pearson correlation with two outliers

```julia-repl
julia> x[3], y[5] = 12, -8
julia> Pingouin.corr(x, y)
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[59]:11
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2    p_val    BF10      power   
     │ Int64  Float64   Float64     Array…                 Float64     Float64   Float64  Float64   Float64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   -0.0410185  [-0.686441, 0.640553]  0.00168252  -0.33109     NaN   0.408345     NaN
```

3. Spearman correlation (robust to outliers)

```julia-repl
julia> Pingouin.corr(x, y, method="spearman")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[68]:15
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2     p_val    BF10     power   
     │ Int64  Float64   Float64     Array…                 Float64     Float64    Float64  Float64  Float64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   -0.0423729  [-0.687157, 0.639752]  0.00179546  -0.330939     NaN      NaN      NaN
```

4. Biweight midcorrelation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="bicor")
1×9 DataFrame
 Row │ n      outliers  r         CI95                  r2         adj_r2     p_val     BF10     power   
     │ Int64  Float64   Float64   Array…                Float64    Float64    Float64   Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   -0.28545  [-0.798245, 0.46725]  0.0814815  -0.224691  0.456539     NaN      NaN
```

5. Percentage bend correlation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="percbend")
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2     p_val     BF10     power   
     │ Int64  Float64   Float64     Array…                 Float64     Float64    Float64   Float64  Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   0.00520452  [-0.661203, 0.667021]  2.70871e-5  -0.333297  0.989398     NaN      NaN
```

6. Shepherd's pi correlation (robust)

```julia-repl
julia> Pingouin.corr(x, y, method="shepherd")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[68]:25
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2     p_val    BF10     power   
     │ Int64  Int64     Float64     Array…                 Float64     Float64    Float64  Float64  Float64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9         0  -0.0423729  [-0.687157, 0.639752]  0.00179546  -0.330939     NaN      NaN      NaN
```

7. One-tailed Pearson correlation

```julia-repl
julia> Pingouin.corr(x, y, tail="one-sided", method="pearson")
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[68]:11
1×9 DataFrame
 Row │ n      outliers  r           CI95                   r2          adj_r2    p_val    BF10      power   
     │ Int64  Float64   Float64     Array…                 Float64     Float64   Float64  Float64   Float64 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     9      NaN   -0.0410185  [-0.686441, 0.640553]  0.00168252  -0.33109     NaN   0.439293     NaN
```
"""
function corr(x::Array{<:Number},
              y::Array{<:Number};
              tail::String="two-sided",
              method::String="pearson")
    # todo: Remove rows with missing values
    # todo: update when p-values available in hypothesistests.jl
    @assert tail in ["two-sided", "one-sided"] "Tail must be \"two-sided\" or \"one-sided\"."

    nx = size(x)[1]
    r = pval = bf10 = outliers = pr = NaN
    # Compute correlation coefficient
    if method == "pearson"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = cor(x, y)
        bf10 = bayesfactor_pearson(r, nx, tail=tail)
    elseif method == "spearman"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = corspearman(x, y)
    elseif method == "kendall"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r = corkendall(x, y)
    elseif method == "bicor"
        r, pval = bicor(x, y)
    elseif method == "percbend"
        r, pval = percbend(x, y)
    elseif method == "shepherd"
        @warn "P-Value not implemented yet in HypothesisTests.jl"
        r, outliers = shepherd(x, y)
        outliers = sum(outliers)
    else
        throw(DomainError(method, "Method not recognized."))
    end

    if r == NaN
        return DataFrame(n=nx,
                         r=NaN,
                         CI95=NaN,
                         r2=NaN,
                         adj_r2=NaN,
                         p_val=NaN,
                         BF10=NaN,
                         power=NaN)
    end

    # Compute r2 and adj_r2
    r2 = r^2
    adj_r2 = 1 - (((1 - r2) * (nx - 1)) / (nx - 3))

    # Compute the parametric 95% confidence interval and power
    ci = compute_esci(stat=r, nx=nx, ny=nx, eftype="r", decimals=6)
    # todo: implement power_corr
    # pr = power_corr(r=r, n=nx, power=None, alpha=0.05, tail=tail)

    return DataFrame(n=nx,
                     outliers=outliers,
                     r=r,
                     CI95=[ci],
                     r2=r2,
                     adj_r2=adj_r2,
                     p_val=pval,
                     BF10=bf10,
                     power=pr)
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
- `tail::String`: Specify whether to return `"one-sided"` or `"two-sided"` p-value. Note that the former are simply half the latter.
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
alternative is to use the :py:func:`pingouin.pcorr` method (see example 4).

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
┌ Warning: P-Value not implemented yet in HypothesisTests.jl
└ @ Main REPL[193]:11
1×9 DataFrame
 Row │ n      outliers  r         CI95                  r2        adj_r2    p_val    BF10     power   
     │ Int64  Float64   Float64   Array…                Float64   Float64   Float64  Float64  Float64 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────
   1 │    30      NaN   0.568169  [0.261409, 0.770684]  0.322816  0.272655     NaN   37.7732     NaN
```

2. Spearman partial correlation with several covariates

```julia-repl
julia> # Partial correlation of x and y controlling for cv1, cv2 and cv3
julia> Pingouin.partial_corr(df, x="x", y="y", covar=["cv1", "cv2", "cv3"], method="spearman")
       n      r         CI95%     r2  adj_r2  p-val  power
spearman  30  0.491  [0.16, 0.72]  0.242   0.185  0.006  0.809
```

3. Semi-partial correlation on x

```julia-repl
julia> pg.partial_corr(data=df, x='x', y='y',
...                 x_covar=['cv1', 'cv2', 'cv3']).round(3)
          n      r         CI95%     r2  adj_r2  p-val   BF10  power
pearson  30  0.463  [0.12, 0.71]  0.215   0.156   0.01  5.404  0.752
```

4. Semi-partial on both x and y controlling for different variables

```julia-repl
julia> pg.partial_corr(data=df, x='x', y='y', x_covar='cv1',
...                 y_covar=['cv2', 'cv3'], method='spearman').round(3)
           n      r         CI95%     r2  adj_r2  p-val  power
spearman  30  0.429  [0.08, 0.68]  0.184   0.123  0.018  0.676
```
"""
function partial_corr(data::DataFrame;
                      x::Union{String,Symbol},
                      y::Union{String,Symbol},
                      covar::Union{Array{T,1},Nothing,String,Symbol}=nothing,
                      x_covar::Union{Array{T,1},Nothing,String,Symbol}=nothing,
                      y_covar::Union{Array{T,1},Nothing,String,Symbol}=nothing,
                      tail::String="two-sided",
                      method::String="pearson")::DataFrame where T <: Union{Nothing,String,Symbol}
    # todo: multiple dispatch
    @assert size(data)[1] > 2 "Data must have at least 3 samples."
    if (covar !== nothing) & (x_covar !== nothing || y_covar !== nothing)
        throw(DomainError([x_covar, y_covar], "Cannot specify both covar and {x,y}_covar."))
    end
    @assert x != covar "x and covar must be independant."
    @assert y != covar "y and covar must be independant."
    @assert x != y "x and y must be independant."
    
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

    data = data[:, col]
    # Remove rows with NaNs
    data = data[completecases(data), :]
    @assert size(data)[1] > 2 "Data must have at least 3 non-NaN samples."

    # Standardize (= no need for an intercept in least-square regression)
    C = (data .- mean.(eachcol(data))') ./ (std.(eachcol(data))')

    if covar[1] !== nothing
        # PARTIAL CORRELATION
        cvar = convert(Matrix, C[:, covar])
        β_x = qr(cvar, Val(true)) \ C[:, x]
        β_y = qr(cvar, Val(true)) \ C[:, y]
        if !isa(β_x, Array)
            β_x = [β_x]
        end
        if !isa(β_y, Array)
            β_y = [β_y]
        end
        res_x = C[:, x] .- cvar .⋅ β_x'  # todo: wrong
        res_y = C[:, y] .- cvar .⋅ β_y'
    else
        # SEMI-PARTIAL CORRELATION
        # Initialize "fake" residuals
        res_x, res_y = data[:, x], data[:, y]
        if x_covar[1] !== nothing
            cvar = convert(Matrix, C[:, x_covar])
            β_x = qr(cvar, Val(true)) \ C[:, x]
            if isa(β_x, Array)
                β_x = β_x[1]
            end
            res_x = C[:, x] .- cvar .⋅ β_x
        end
        if y_covar[1] !== nothing
            cvar = convert(Matrix, C[:, y_covar])
            β_y = qr(cvar, Val(true)) \ C[:, y]
            if isa(β_y, Array)
                β_y = β_y[1]
            end
            res_y = C[:, y] .- cvar .⋅ β_y
        end
    end

    return corr(res_x[:, 1], res_y[:, 1], method=method, tail=tail)
end
