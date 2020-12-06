using DataFrames
using Distributions
using HypothesisTests
using StatsBase

include("effsize.jl")

"""
    madmedianrule(a)

Robust outlier detection based on the MAD-median rule.

Arguments
----------
- `a::Array{<:Number}`: Input array. Must be one-dimensional.

Returns
-------
- `outliers::Array{Bool}`: Boolean array indicating whether each sample is an outlier (true) or not (false).

See also
--------
`Statistics.mad`

Notes
-----
The MAD-median-rule ([1], [2]) will refer to declaring \$X_i\$
an outlier if

\$\\frac{\\left | X_i - M \\right |}{\\text{MAD}_{\\text{norm}}} > K\$,

where \$M\$ is the median of \$X\$,
\$\\text{MAD}_{\\text{norm}}\$ the normalized median absolute deviation
of \$X\$, and \$K\$ is the square
root of the .975 quantile of a \$X^2\$ distribution with one degree
of freedom, which is roughly equal to 2.24.

References
----------
[1] Hall, P., Welsh, A.H., 1985. Limit theorems for the median
deviation. Ann. Inst. Stat. Math. 37, 27–36.
https://doi.org/10.1007/BF02481078

[2] Wilcox, R. R. Introduction to Robust Estimation and Hypothesis
Testing. (Academic Press, 2011).

Examples
--------
```julia-repl
julia> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
julia> Pingouin.madmedianrule(a)
8-element Array{Bool,1}:
 0
 0
 0
 0
 0
 1
 0
 0
```
"""
function madmedianrule(a::Array{<:Number})::Array{Bool}
    @assert length(size(a)) == 1 "Only 1D array / list are supported for this function."
    k = sqrt(quantile(Chisq(1), 0.975))
    return (abs.(a .- median(a)) ./ mad(a)) .> k
end


"""
    mwu(x, y)

Mann-Whitney U Test (= Wilcoxon rank-sum test). It is the non-parametric
version of the independent T-test.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. `x` and `y` must be independent.

Returns
-------
- `stats::DataFrame`
    * `'U-val'`: U-value
    * `'p-val'`: p-value
    * `'RBC'`: rank-biserial correlation
    * `'CLES'`: common language effect size

See also
--------
- `HypothesisTests.MannWhitneyUTest`,
- [`wilcoxon`](@ref),
- [`ttest`](@ref).

Notes
-----
The Mann–Whitney U test [1], (also called Wilcoxon rank-sum test) is a
non-parametric test of the null hypothesis that it is equally likely that
a randomly selected value from one sample will be less than or greater
than a randomly selected value from a second sample. The test assumes
that the two samples are independent. This test corrects for ties and by
default uses a continuity correction
(see `HypothesisTests.MannWhitneyUTest` for details).

The rank biserial correlation [2] is the difference between
the proportion of favorable evidence minus the proportion of unfavorable
evidence.

The common language effect size is the proportion of pairs where \$x\$ is
higher than \$y\$. It was first introduced by McGraw and Wong (1992) [3].
Pingouin uses a brute-force version of the formula given by Vargha and
Delaney 2000 [4]:

\$\\text{CL} = P(X > Y) + .5 \\times P(X = Y)\$

The advantage is of this method are twofold. First, the brute-force
approach pairs each observation of \$x\$ to its \$y\$ counterpart, and
therefore does not require normally distributed data. Second, the formula
takes ties into account and therefore works with ordinal data.

References
----------
[1] Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of
two random variables is stochastically larger than the other.
The annals of mathematical statistics, 50-60.

[2] Kerby, D. S. (2014). The simple difference formula: An approach to
teaching nonparametric correlation. Comprehensive Psychology,
3, 11-IT.

[3] McGraw, K. O., & Wong, S. P. (1992). A common language effect size
statistic. Psychological bulletin, 111(2), 361.

[4] Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of
the “CL” Common Language Effect Size Statistics of McGraw and Wong.
Journal of Educational and Behavioral Statistics: A Quarterly
Publication Sponsored by the American Educational Research
Association and the American Statistical Association, 25(2),
101–132. https://doi.org/10.2307/1165329

Examples
--------
```julia-repl
julia> x = [1,4,2,5,3,6,9,8,7]
julia> y = [2,4,1,5,10,1,4,9,8,5]
julia> Pingouin.mwu(x, y)
1×4 DataFrame
│ Row │ U_val   │ p_val    │ RBC        │ CLES     │
│     │ Float64 │ Float64  │ Float64    │ Float64  │
├─────┼─────────┼──────────┼────────────┼──────────┤
│ 1   │ 46.5    │ 0.934494 │ -0.0333333 │ 0.516667 │
```

Compare with HypothesisTests

```julia-repl
julia> using HypothesisTests
julia> MannWhitneyUTest(x, y)
Approximate Mann-Whitney U test
-------------------------------
Population details:
    parameter of interest:   Location parameter (pseudomedian)
    value under h_0:         0
    point estimate:          0.5

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.9345

Details:
    number of observations in each group: [9, 10]
    Mann-Whitney-U statistic:             46.5
    rank sums:                            [91.5, 98.5]
    adjustment for ties:                  90.0
    normal approximation (μ, σ):          (1.5, 12.1666)
```
"""
function mwu(x::Array{<:Number}, y::Array{<:Number})::DataFrame
    # todo: remove na
    # possible_tails = ["two-sided", "one-sided", "greater", "less"]
    # @assert tail in possible_tails "Invalid tail argument."

    # if tail == "one-sided"
        # Detect the direction of the test based on the median
        # tail = median(x) < median(y) ? "less" : "greater"
    # end

    results = MannWhitneyUTest(x, y)
    uval, pval = results.U, pvalue(results)
    
    # Effect size 1: Common Language Effect Size
    cles = compute_effsize(x, y, eftype="cles")

    # Effect size 2: Rank Biserial Correlation (Wendt, 1972)
    rbc = 1 - (2 * uval) / (length(x) * length(y))

    return DataFrame(U_val=uval, p_val=pval, RBC=rbc, CLES=cles)
end


"""
    wilcoxon(x, y)

Wilcoxon signed-rank test. It is the non-parametric version of the
paired T-test.

Arguments
----------
- `x, y::Array{<:Number}`: First and second set of observations. ``x`` and ``y`` must be related (e.g repeated measures) and, therefore, have the same number of samples. Note that a listwise deletion of missing values is automatically applied.

Returns
-------
- `stats::DataFrame`
    * `'W-val'`: W-value
    * `'p-val'`: p-value
    * `'RBC'`: matched pairs rank-biserial correlation (effect size)
    * `'CLES'`: common language effect size

See also
--------
- `HypothesisTests.SignedRankTest`,
- [`mwu`](@ref).

Notes
-----
The Wilcoxon signed-rank test [1] tests the null hypothesis that two
related paired samples come from the same distribution. In particular,
it tests whether the distribution of the differences \$x - y\$ is symmetric
about zero. A continuity correction is applied by default
(see `HypothesisTests.SignedRankTest` for details).

The matched pairs rank biserial correlation [2] is the simple difference
between the proportion of favorable and unfavorable evidence; in the case
of the Wilcoxon signed-rank test, the evidence consists of rank sums
(Kerby 2014):

\$r = f - u\$

The common language effect size is the proportion of pairs where ``x`` is
higher than ``y``. It was first introduced by McGraw and Wong (1992) [3].
Pingouin uses a brute-force version of the formula given by Vargha and
Delaney 2000 [4]:

\$\\text{CL} = P(X > Y) + .5 \\times P(X = Y)\$

The advantage is of this method are twofold. First, the brute-force
approach pairs each observation of ``x`` to its ``y`` counterpart, and
therefore does not require normally distributed data. Second, the formula
takes ties into account and therefore works with ordinal data.

References
----------
[1] Wilcoxon, F. (1945). Individual comparisons by ranking methods.
Biometrics bulletin, 1(6), 80-83.

[2] Kerby, D. S. (2014). The simple difference formula: An approach to
teaching nonparametric correlation. Comprehensive Psychology,
3, 11-IT.

[3] McGraw, K. O., & Wong, S. P. (1992). A common language effect size
statistic. Psychological bulletin, 111(2), 361.

[4] Vargha, A., & Delaney, H. D. (2000). A Critique and Improvement of
the “CL” Common Language Effect Size Statistics of McGraw and Wong.
Journal of Educational and Behavioral Statistics: A Quarterly
Publication Sponsored by the American Educational Research
Association and the American Statistical Association, 25(2),
101–132. https://doi.org/10.2307/1165329

Examples
--------
Wilcoxon test on two related samples.

```julia-repl
julia> x = [20, 22, 19, 20, 22, 18, 24, 20, 19, 24, 26, 13]
julia> y = [38, 37, 33, 29, 14, 12, 20, 22, 17, 25, 26, 16]
julia> Pingouin.wilcoxon(x, y)
1×4 DataFrame
│ Row │ W_val   │ p_val    │ RBC       │ CLES     │
│     │ Float64 │ Float64  │ Float64   │ Float64  │
├─────┼─────────┼──────────┼───────────┼──────────┤
│ 1   │ 20.5    │ 0.288086 │ -0.378788 │ 0.395833 │
```

Compare with HypothesisTests

```julia-repl
julia> using HypothesisTests
julia> SignedRankTest(x, y)
Exact Wilcoxon signed rank test
-------------------------------
Population details:
    parameter of interest:   Location parameter (pseudomedian)
    value under h_0:         0
    point estimate:          -1.5
    95% confidence interval: (-9.0, 2.5)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:           0.2881

Details:
    number of observations:      12
    Wilcoxon rank-sum statistic: 20.5
    rank sums:                   [20.5, 45.5]
    adjustment for ties:         6.0
```
"""
function wilcoxon(x::Array{<:Number}, y::Array{<:Number})::DataFrame
    # todo: remove na
    results = SignedRankTest(x, y)

    wval, pval = results.W, pvalue(results)

    # Effect size 1: Common Language Effect Size
    cles = compute_effsize(x, y, eftype="cles")

    # Effect size 2: matched-pairs rank biserial correlation (Kerby 2014)
    d = x .- y
    d = d[d .!= 0]
    r = tiedrank(abs.(d))
    rsum = sum(r)
    r_plus = sum((d .> 0) .* r)
    r_minus = sum((d .< 0) .* r)
    rbc = r_plus / rsum - r_minus / rsum

    return DataFrame(W_val=wval, p_val=pval, RBC=rbc, CLES=cles)
end


"""
    kruskal(data[, dv, between, detailed])

Kruskal-Wallis H-test for independent samples.

Arguments
----------
- `data::DataFrame`: DataFrame,
- `dv::String`: Name of column containing the dependent variable,
- `between::String`: Name of column containing the between factor.

Returns
-------
- `stats::DataFrame`
    * `'H'`: The Kruskal-Wallis H statistic, corrected for ties,
    * `'p-unc'`: Uncorrected p-value,
    * `'dof'`: degrees of freedom.

Notes
-----
The Kruskal-Wallis H-test tests the null hypothesis that the population
median of all of the groups are equal. It is a non-parametric version of
ANOVA. The test works on 2 or more independent samples, which may have
different sizes.

Due to the assumption that H has a chi square distribution, the number of
samples in each group must not be too small. A typical rule is that each
sample must have at least 5 measurements.

NaN values are automatically removed.

Examples
--------
Compute the Kruskal-Wallis H-test for independent samples.

```julia-repl
julia> data = Pingouin.read_dataset("anova")
julia> Pingouin.kruskal(data, dv="Pain threshold", between="Hair color")
1×4 DataFrame
│ Row │ Source     │ ddof  │ H       │ p_unc     │
│     │ String     │ Int64 │ Float64 │ Float64   │
├─────┼────────────┼───────┼─────────┼───────────┤
│ 1   │ Hair color │ 3     │ 10.5886 │ 0.0141716 │
```
"""
function kruskal(data;
                 dv::Union{String,Symbol},
                 between::Union{String,Symbol})::DataFrame
    # todo: remove nans

    group_names = unique(data[between])
    groups = [data[data[between] .== v, dv] for v in group_names]

    results = KruskalWallisTest(groups...)

    ddof = results.df
    H = results.H
    p_unc = pvalue(results)

    return DataFrame(Source=between,
                     ddof=ddof,
                     H=H,
                     p_unc=p_unc)
end


"""
    cochran(data[, dv, within, subject])

Cochran Q test. A special case of the Friedman test when the dependent
variable is binary.

Arguments
----------
- `data::DataFrame`
- `dv::Union{Nothing,String,Symbol}`: Name of column containing the binary dependent variable.
- `within::Union{Nothing,String,Symbol}`: Name of column containing the within-subject factor.
- `subject::Union{Nothing,String,Symbol}`: Name of column containing the subject identifier.

Returns
-------
- `stats::DataFrame`
    * `'Q'`: The Cochran Q statistic,
    * `'p-unc'`: Uncorrected p-value,
    * `'ddof'`: degrees of freedom.

Notes
-----
The Cochran Q test [1] is a non-parametric test for ANOVA with repeated
measures where the dependent variable is binary.

Data are expected to be in long-format. NaN are automatically removed
from the data.

The Q statistics is defined as:

\$Q = \\frac{(r-1)(r\\sum_j^rx_j^2-N^2)}{rN-\\sum_i^nx_i^2}\$

where ``N`` is the total sum of all observations, ``j=1,...,r``
where ``r`` is the number of repeated measures, ``i=1,...,n`` where
``n`` is the number of observations per condition.

The p-value is then approximated using a chi-square distribution with
``r-1`` degrees of freedom:

\$Q \\sim \\chi^2(r-1)\$

References
----------
[1] Cochran, W.G., 1950. The comparison of percentages in matched
samples. Biometrika 37, 256–266.
https://doi.org/10.1093/biomet/37.3-4.256

Examples
--------
Compute the Cochran Q test for repeated measurements.

```julia-repl
julia> data = Pingouin.read_dataset("cochran");
julia> cochran(data, dv="Energetic", within="Time", subject="Subject")
1×4 DataFrame
│ Row │ Source │ ddof  │ Q       │ p_unc     │
│     │ String │ Int64 │ Float64 │ Float64   │
├─────┼────────┼───────┼─────────┼───────────┤
│ 1   │ Time   │ 2     │ 6.70588 │ 0.0349813 │
```
"""
function cochran(data::DataFrame;
                 dv::Union{String,Symbol},
                 within::Union{String,Symbol},
                 subject::Union{String,Symbol})::DataFrame
    # todo: remove na
    grp = combine(groupby(data, within), dv=>sum=>dv)[dv]
    grp_s = combine(groupby(data, subject), dv=>sum=>dv)[dv]
    k = length(Set(data[:, within]))
    ddof = k - 1

    # Q statistic and p-value
    q = (ddof * (k * sum(grp.^2) - sum(grp).^2)) / (k * sum(grp) - sum(grp_s.^2))
    p_unc = ccdf(Chisq(ddof), q)

    return DataFrame(Source = within,
                     ddof = ddof,
                     Q = q,
                     p_unc = p_unc)
end


"""
    harrelldavis(x[, q, dim])

*EXPERIMENTAL* Harrell-Davis robust estimate of the ``q^{th}`` quantile(s) of the
data. *TESTS NEEDED*

Arguments
----------
- `x::Array{<:Number}`: Data, must be a one or two-dimensional vector.
- `q::Union{Float64,Array{Float64}}`: Quantile or sequence of quantiles to compute, must be between 0 and 1. Default is ``0.5``.
- `dim::Int64`: Axis along which the MAD is computed. Default is the first axis. Can be either 1 or 2.

Returns
-------
- `y::Union{Float64,Array{Float64}}`: The estimated quantile(s). If `quantile` is a single quantile, will return a float, otherwise will compute each quantile separately and returns an array of floats.

Notes
-----
The Harrell-Davis method [1] estimates the ``q^{th}`` quantile by a
linear combination of  the  order statistics. Results have been tested
against a Matlab implementation [2]. Note that this method is also
used to measure the confidence intervals of the difference between
quantiles of two groups, as implemented in the shift function [3].

See Also
--------
[`plot_shift`](@ref)

References
----------
[1] Frank E. Harrell, C. E. Davis, A new distribution-free quantile
estimator, Biometrika, Volume 69, Issue 3, December 1982, Pages
635–640, https://doi.org/10.1093/biomet/69.3.635

[2] https://github.com/GRousselet/matlab_stats/blob/master/hd.m

[3] Rousselet, G. A., Pernet, C. R. and Wilcox, R. R. (2017). Beyond
differences in means: robust graphical methods to compare two groups
in neuroscience. Eur J Neurosci, 46: 1738-1748.
https://doi.org/doi:10.1111/ejn.13610

Examples
--------
Estimate the 0.5 quantile (i.e median) of 100 observation picked from a
normal distribution with zero mean and unit variance.

```julia-repl
julia> using Distributions, Random
julia> d = Normal(0, 1)
julia> x = rand(d, 100);
>>> Pingouin.harrelldavis(x, 0.5)
-0.3197175569523778
```

Several quantiles at once

```julia-repl
julia> Pingouin.harrelldavis(x, [0.25, 0.5, 0.75])
3-element Array{Float64,1}:
 -0.8584761447019648
 -0.3197175569523778
  0.30049291160713604
```

On the last axis of a 2D vector (default)


```julia-repl
julia> using Distributions, Random
julia> d = Normal(0, 1)
julia> x = rand(d, (100, 100));
julia> Pingouin.harrelldavis(x, 0.5)
100×1 Array{Float64,2}:
  0.08776830864191214
  0.03470963005927001
 -0.0805646920967012
  0.3314919956251108
  0.3111971350475172
  ⋮
  0.10769293112437549
 -0.10622118136247076
 -0.13230506142402296
 -0.09693123033727057
 -0.2135938540892071
```

On the first axis

```julia-repl
julia> Pingouin.harrelldavis(x, 0.5, 1)
1×100 Array{Float64,2}:
 0.0112259  -0.0409635  -0.0918462 ...
```

On the first axis with multiple quantiles

```julia-repl
julia> Pingouin.harrelldavis(x, [0.5, 0.75], 1)
1×100 Array{Float64,2}:
 0.0112259  -0.0409635  -0.0918462 ...
```
"""
function harrelldavis(x::Array{T,1} where T<:Number,
                     q::Float64=0.5)
    sort!(x)

    n = length(x)
    vec = convert(Array, range(1, n, step=1))

    # Harrell-Davis estimate of the qth quantile
    m1 = (n + 1) * q
    m2 = (n + 1) * (1 - q)
    w = cdf(Beta(m1, m2), vec ./ n) - cdf(Beta(m1, m2), (vec .- 1) ./ n)

    return sum(w .* x)
end
function harrelldavis(x::Array{T,1} where T<:Number,
                      q::Array{Float64}=[0.5, 0.75])
    y = Array{Float64}(undef, length(q))
    for (i, _q) in enumerate(q)
        y[i] = harrelldavis(x, _q)
    end
    return y
end
function harrelldavis(x::Array{T,2} where T<:Number,
                      q::Float64=0.5,
                      dim::Int64=2)
    @assert dim in [1, 2]

    sort!(x, dims=dim)

    n = size(x)[dim]
    vec = convert(Array, range(1, n, step=1))

    # Harrell-Davis estimate of the qth quantile
    m1 = (n + 1) * q
    m2 = (n + 1) * (1 - q)
    w = cdf.(Beta(m1, m2), vec ./ n) - cdf.(Beta(m1, m2), (vec .- 1) ./ n)

    # todo: triple check transpose
    if dim == 1
        return sum(w .* x, dims=dim)
    elseif dim == 2
        return sum(transpose(w) .* x, dims=dim)
    end
end
function harrelldavis(x::Array{T,2} where T<:Number,
                     q::Array{Float64}=[0.5, 0.75],
                     dim::Int64=1)
    y = Array{Array}(undef,length(q))
    for (i, _q) in enumerate(q)
        y[i] = harrelldavis(x, _q, dim)
    end
    return y
end


"""
    friedman(data, dv, within, subject)

Friedman test for repeated measurements.

Arguments
----------
- `data::DataFrame`,
- `dv::Union{String,Symbol}`: Name of column containing the dependent variable,
- `within::Union{String,Symbol}`: Name of column containing the within-subject factor,
- `subject::Union{String,Symbol}`: Name of column containing the subject identifier.

Returns
-------
- `stats::DataFrame`
    * `"Q"`: The Friedman Q statistic, corrected for ties,
    * `"p-unc"`: Uncorrected p-value,
    * `"ddof"`: degrees of freedom.

Notes
-----
The Friedman test is used for one-way repeated measures ANOVA by ranks.

Data are expected to be in long-format.

Note that if the dataset contains one or more other within subject
factors, an automatic collapsing to the mean is applied on the dependent
variable (same behavior as the ezANOVA R package). As such, results can
differ from those of JASP. If you can, always double-check the results.

Due to the assumption that the test statistic has a chi squared
distribution, the p-value is only reliable for n > 10 and more than 6
repeated measurements.

NaN values are automatically removed.

Examples
--------
Compute the Friedman test for repeated measurements.

```julia-repl
julia> data = Pingouin.read_dataset("rm_anova")
julia> Pingouin.friedman(data,
                         dv="DesireToKill",
                         within="Disgustingness",
                         subject="Subject")
                         1×4 DataFrame
1×4 DataFrame
 Row │ Source          ddof   Q        p_unc      
     │ String          Int64  Float64  Float64    
─────┼────────────────────────────────────────────
   1 │ Disgustingness      1  9.22785  0.00238362
```
"""
function friedman(data::DataFrame;
                  dv::Union{String,Symbol},
                  within::Union{String,Symbol},
                  subject)::DataFrame
    # Collapse to the mean
    function m(x)
        return mean(skipmissing(x))
    end
    data = combine(groupby(data, [subject, within]), dv=>m=>dv)

    # Extract number of groups and total sample size
    grp = groupby(data, within)
    rm = unique(data[!, within])
    k = length(rm)
    X = hcat([convert(Array{Float64}, g[!, dv]) for g in grp]...)
    n = size(X)[1]

    # Rank per subject
    ranked = Array{Float64, 2}(undef, size(X))
    for i in 1:n
        ranked[i, :] = tiedrank(X[i, :])
    end
    
    ssbn = sum(sum(ranked, dims=1).^2)

    # Compute the test statistic
    Q = (12 / (n * k * (k + 1))) * ssbn - 3 * n * (k + 1)

    # Correct for ties
    ties = 0
    for i in 1:n
        repnum = values(countmap(X[i, :]))
        for t in repnum
            ties += t * (t * t - 1)
        end
    end

    c = 1 - ties / float(k * (k * k - 1) * n)
    Q /= c

    # Approximate the p-value
    ddof1 = k - 1
    p_unc = ccdf(Chisq(ddof1), Q)

    return DataFrame(Source = within,
                     ddof = ddof1,
                     Q = Q,
                     p_unc = p_unc)
end
