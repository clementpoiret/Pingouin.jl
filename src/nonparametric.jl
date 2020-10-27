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