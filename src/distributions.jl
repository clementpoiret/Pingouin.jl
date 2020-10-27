using DataFrames
using Distributions
using HypothesisTests
using LinearAlgebra
using StatsBase

include("_shapiro.jl")
include("_homoscedasticity.jl")

"""
    gzscore(x)

Geometric standard (Z) score.

Arguments
----------
- `x::Array{<:Number}`: Array of raw values

Returns
-------
- `gzscore::Array{<:Number}`: Array of geometric z-scores (same shape as x)

Notes
-----
Geometric Z-scores are better measures of dispersion than arithmetic
z-scores when the sample data come from a log-normally distributed
population [1].

Given the raw scores \$x\$, the geometric mean \$\\mu_g\$ and
the geometric standard deviation \$\\sigma_g\$,
the standard score is given by the formula:

\$z = \\frac{log(x) - log(\\mu_g)}{log(\\sigma_g)}\$

References
----------
[1] https://en.wikipedia.org/wiki/Geometric_standard_deviation

Examples
--------
Standardize a lognormal-distributed vector:

```julia-repl
julia> raw = [1,4,5,4,1,2,5,8,6,6,9,8,3]
julia> z = Pingouin.gzscore(raw)
13-element Array{Float64,1}:
 -1.8599725059104346
  0.03137685347921089
  0.3358161014965816
  0.03137685347921089
 -1.8599725059104346
  ⋮
  0.5845610789821727
  0.5845610789821727
  1.1377453044851344
  0.9770515331740336
 -0.3611136007126501
```
"""
function gzscore(x::Array{<:Number})::Array{<:Number}
    # Geometric mean
    geo_mean = geomean(x)
    # Geometric standard deviation
    gstd = exp(sqrt(sum((@. log(x ./ geo_mean)).^2) / (length(x) - 1)))
    # Geometric z-score
    return @. log(x ./ geo_mean) ./ log(gstd)
end


"""
    anderson(x[, dist])

Anderson-Darling test of distribution.

Arguments
----------
- `x::Array{<:Number}`: Array of sample data. May be different lengths,
- `dist::Union{String, Distribution}`: Distribution ("norm", "expon", "logistic", "gumbel").

Returns
-------
- `H::Bool`: True if data comes from this distribution,
- `P::Float64`: The significance levels for the corresponding critical values in %.
(See :`HypothesisTests.OneSampleADTest` for more details).

Examples
--------
1. Test that an array comes from a normal distribution

```julia-repl
julia> x = [2.3, 5.1, 4.3, 2.6, 7.8, 9.2, 1.4]
julia> Pingouin.anderson(x, dist="norm")
(false, 8.571428568870942e-5)
```

2. Test that an array comes from a custom distribution

```julia-repl
> x = [2.3, 5.1, 4.3, 2.6, 7.8, 9.2, 1.4]
> Pingouin.anderson(x, dist=Normal(1,5))
(false, 0.04755873570126501)
```
"""
function anderson(x::Array{<:Number};
                  dist::Union{String, Distribution}=Normal(),
                  α::Float64=0.05)::Tuple{Bool, Float64}
    # todo: implement support for multiple samples
    if isa(dist, String)
        if dist == "norm"
            dist = Normal()
        elseif dist == "expon"
            dist = Exponential()
        elseif dist == "logistic"
            dist = Logistic()
        elseif dist == "gumbel"
            dist = Gumbel()
        end
    end

    andersontest = OneSampleADTest(x, dist)
    P = pvalue(andersontest)
    H = P < α

    return !H, P
end


"""
    normality(data[, dv, group, method, α])

Univariate normality test.

Arguments
----------
- `data`: Iterable. Can be either a single list, 1D array, or a wide- or long-format dataframe.
- `dv::Union{Symbol, String, Nothing}`: Dependent variable (only when ``data`` is a long-format dataframe).
- `group::Union{Symbol, String, Nothing}`: Grouping variable (only when ``data`` is a long-format dataframe).
- `method::String`: Normality test. `'shapiro'` (default) performs the Shapiro-Wilk test using the AS R94 algorithm. If the kurtosis is higher than 3, it  performs a Shapiro-Francia test for leptokurtic distributions. Supported values: ["shapiro", "jarque_bera"].
- `α::Float64`: Significance level.

Returns
-------
- `stats::DataFrame`:
    * `W`: Test statistic,
    * `pval`: p-value,
    * `normal`: true if `data` is normally distributed.

See Also
--------
- [`homoscedasticity`](@ref): Test equality of variance.
- [`sphericity`](@ref): Mauchly's test for sphericity.

Notes
-----
The Shapiro-Wilk test calculates a :math:`W` statistic that tests whether a
random sample \$x_1, x_2, ..., x_n\$ comes from a normal distribution.

The \$W\$ is normalized (\$W = (W - μ) / σ\$)

The null-hypothesis of this test is that the population is normally
distributed. Thus, if the p-value is less than the
chosen alpha level (typically set at 0.05), then the null hypothesis is
rejected and there is evidence that the data tested are not normally
distributed.

The result of the Shapiro-Wilk test should be interpreted with caution in
the case of large sample sizes (>5000). Indeed, quoting from
[Wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test):

*"Like most statistical significance tests, if the sample size is
sufficiently large this test may detect even trivial departures from
the null hypothesis (i.e., although there may be some statistically
significant effect, it may be too small to be of any practical
significance); thus, additional investigation of the effect size is
typically advisable, e.g., a Q–Q plot in this case."*

The Jarque-Bera statistic is to test the null hypothesis that a real-valued vector \$y\$
is normally distributed. Note that the approximation by the Chi-squared distribution does
not work well and the speed of convergence is slow.
In small samples, the test tends to be over-sized for nominal levels up to about 3% and
under-sized for larger nominal levels (Mantalos, 2010).

Note that missing values are automatically removed (casewise deletion).

References
----------
* Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
for normality (complete samples). Biometrika, 52(3/4), 591-611.

* Panagiotis Mantalos, 2011, "The three different measures of the sample skewness and
kurtosis and the effects to the Jarque-Bera test for normality", International Journal
of Computational Economics and Econometrics, Vol. 2, No. 1,
[link](http://dx.doi.org/10.1504/IJCEE.2011.040576).

* https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

* [Jarque-Bera test on Wikipedia](https://en.wikipedia.org/wiki/Jarque–Bera_test)

Examples
--------
1. Shapiro-Wilk test on a 1D array

```julia-repl
julia> dataset = Pingouin.read_dataset("anova")
julia> Pingouin.normality(dataset["Pain threshold"])
1×3 DataFrame
│ Row │ W         │ pval     │ normal │
│     │ Float64   │ Float64  │ Bool   │
├─────┼───────────┼──────────┼────────┤
│ 1   │ -0.842541 │ 0.800257 │ 1      │
```

2. Wide-format dataframe using Jarque-Bera test

```julia-repl
julia> dataset = Pingouin.read_dataset("mediation")
julia> Pingouin.normality(dataset, method="jarque_bera")
│ Row │ dv     │ W        │ pval        │ normal │
│     │ Symbol │ Float64  │ Float64     │ Bool   │
├─────┼────────┼──────────┼─────────────┼────────┤
│ 1   │ X      │ 1.42418  │ 0.490618    │ 1      │
│ 2   │ M      │ 0.645823 │ 0.724038    │ 1      │
│ 3   │ Y      │ 0.261805 │ 0.877303    │ 1      │
│ 4   │ Mbin   │ 16.6735  │ 0.000239553 │ 0      │
│ 5   │ Ybin   │ 16.6675  │ 0.000240265 │ 0      │
│ 6   │ W1     │ 5.40923  │ 0.0668961   │ 1      │
│ 7   │ W2     │ 80.6857  │ 3.01529e-18 │ 0      │
```

3. Long-format dataframe

```julia-repl
julia> dataset = Pingouin.read_dataset("rm_anova2")
julia> Pingouin.normality(dataset, dv=:Performance, group=:Time)
│ Row │ Time   │ W         │ pval      │ normal │
│     │ String │ Float64   │ Float64   │ Bool   │
├─────┼────────┼───────────┼───────────┼────────┤
│ 1   │ Pre    │ 0.0532374 │ 0.478771  │ 1      │
│ 2   │ Post   │ 1.30965   │ 0.0951576 │ 1      │
```
"""
function normality(data;
                   dv::Union{Symbol, String, Nothing}=nothing,
                   group::Union{Symbol, String, Nothing}=nothing,
                   method::String="shapiro",
                   α::Float64=0.05)::DataFrame
    func = eval(Meta.parse(method))
    if isa(data, Array{})
        return func(data, α)
    else 
        if dv === nothing && group === nothing
            # wide dataframe
            numdata = data[ :, colwise(x -> (eltype(x) <: Number), data)]

            result = DataFrame()
            for column in propertynames(numdata)
                r = func(numdata[column], α)
                insertcols!(r, 1, :dv => column)
                append!(result, r)
            end

            return result
        else
            # long dataframe
            group = Symbol(group)
            dv = Symbol(dv)

            @assert group in propertynames(data)
            @assert dv in propertynames(data)
            
            grp = groupby(data, group, sort=false)
            result = DataFrame()
            for subdf in grp
                r = func(DataFrame(subdf)[dv], α)
                insertcols!(r, 1, group => subdf[1, group])

                append!(result, r)
            end
            
            return result
            
        end
    end
end


function shapiro(x::Array{<:Number}, α::Float64=0.05)::DataFrame
    """
    Compute the Shapiro-Wilk statistic to test the null hypothesis that a real-valued vector \$y\$ is normally distributed.
    """
    x = x[@. !isnan.(x)]
    
    n = length(x)

    if n <= 3
        throw(DomainError(x, "Data must be at least of length 4."))
    end

    if n >= 5000
    print("[WARN] x contains more than 5000 samples. The test might be incorrect.")
    end

    if minimum(x) == maximum(x)
        throw(DomainError(x, "All values are identical."))
    end

    H, SW, P = shapiro_wilk(x, α)

    return DataFrame(W=SW, pval=P, normal=!H)
end


function jarque_bera(x::Array{<:Number}, α::Float64=0.05)::DataFrame
    """
    Compute the Jarque-Bera statistic to test the null hypothesis that a real-valued vector \$y\$ is normally distributed.
    """
    test = JarqueBeraTest(x)

    JB = test.JB
    P = pvalue(test)
    H = (α >= P)

    return DataFrame(W=JB, pval=P, normal=!H)
end


"""
    homoscedasticity(data[, dv, group, method, α])

Test equality of variance.

Arguments
----------
- `data`: Iterable. Can be either an Array iterables or a wide- or long-format dataframe.
- `dv::Union{Symbol, String, Nothing}`: Dependent variable (only when ``data`` is a long-format dataframe).
- `group::Union{Symbol, String, Nothing}`: Grouping variable (only when ``data`` is a long-format dataframe).
- `method::String`: Statistical test. `'levene'` (default) performs the Levene test and `'bartlett'` performs the Bartlett test. The former is more robust to departure from normality.
- `α::Float64`: Significance level.

Returns
-------
- `stats::DataFrame`:
    * `W/T`: Test statistic ('W' for Levene, 'T' for Bartlett), 
    * `pval`: p-value,
    * `equal_var`: true if `data` has equal variance.

See Also
--------
- [`normality`](@ref) : Univariate normality test.
- [`sphericity`](@ref) : Mauchly's test for sphericity.

Notes
-----
The **Bartlett** \$T\$ statistic [1] is defined as:

\$T = \\frac{(N-k) \\ln{s^{2}_{p}} - \\sum_{i=1}^{k}(N_{i} - 1) \\ln{s^{2}_{i}}}{1 + (1/(3(k-1)))((\\sum_{i=1}^{k}{1/(N_{i} - 1))} - 1/(N-k))}\$

where \$s_i^2\$ is the variance of the \$i^{th}\$ group,
\$N\$ is the total sample size, \$N_i\$ is the sample size of the
\$i^{th}\$ group, \$k\$ is the number of groups,
and \$s_p^2\$ is the pooled variance.

The pooled variance is a weighted average of the group variances and is
defined as:

\$s^{2}_{p} = \\sum_{i=1}^{k}(N_{i} - 1)s^{2}_{i}/(N-k)\$

The p-value is then computed using a chi-square distribution:

\$T \\sim \\chi^2(k-1)\$

The **Levene** \$W\$ statistic [2] is defined as:

\$W = \\frac{(N-k)} {(k-1)} \\frac{\\sum_{i=1}^{k}N_{i}(\\overline{Z}_{i.}-\\overline{Z})^{2} } {\\sum_{i=1}^{k}\\sum_{j=1}^{N_i}(Z_{ij}-\\overline{Z}_{i.})^{2} }\$

where \$Z_{ij} = |Y_{ij} - \\text{median}({Y}_{i.})|\$,
\$\\overline{Z}_{i.}\$ are the group means of \$Z_{ij}\$ and
\$\\overline{Z}\$ is the grand mean of \$Z_{ij}\$.

The p-value is then computed using a F-distribution:

\$W \\sim F(k-1, N-k)\$

**WARNING:** Missing values are not supported for this function.
Make sure to remove them before.

References
----------
[1] Bartlett, M. S. (1937). Properties of sufficiency and statistical
tests. Proc. R. Soc. Lond. A, 160(901), 268-282.

[2] Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the
equality of variances. Journal of the American Statistical
Association, 69(346), 364-367.

Examples
--------
1. Levene test on a wide-format dataframe

```julia-repl
julia> data = Pingouin.read_dataset("mediation")
julia> Pingouin.homoscedasticity(data[["X", "Y", "M"]])
1×3 DataFrame
│ Row │ W       │ pval     │ equal_var │
│     │ Float64 │ Float64  │ Bool      │
├─────┼─────────┼──────────┼───────────┤
│ 1   │ 1.17352 │ 0.310707 │ 1         │
```

2. Bartlett test using an array of arrays

```julia-repl
julia> data = [[4, 8, 9, 20, 14], [5, 8, 15, 45, 12]]
julia> Pingouin.homoscedasticity(data, method="bartlett", α=.05)
1×3 DataFrame
│ Row │ T       │ pval     │ equal_var │
│     │ Float64 │ Float64  │ Bool      │
├─────┼─────────┼──────────┼───────────┤
│ 1   │ 2.87357 │ 0.090045 │ 1         │
```

3. Long-format dataframe

```julia-repl
julia> data = Pingouin.read_dataset("rm_anova2")
julia> Pingouin.homoscedasticity(data, dv="Performance", group="Time")
1×3 DataFrame
│ Row │ W       │ pval      │ equal_var │
│     │ Float64 │ Float64   │ Bool      │
├─────┼─────────┼───────────┼───────────┤
│ 1   │ 3.1922  │ 0.0792169 │ 1         │
```
"""
function homoscedasticity(data;
                          dv::Union{Symbol, String, Nothing}=nothing,
                          group::Union{Symbol, String, Nothing}=nothing,
                          method::String="levene",
                          α::Float64=0.05)::DataFrame
    @assert method in ["levene", "bartlett"]
    func = eval(Meta.parse(method))

    if isa(data, Array{})
        H, W, P = func(data, α=α)
        if method == "levene"
            return DataFrame(W=W, pval=P, equal_var=!H)
        elseif method == "bartlett"
            return DataFrame(T=W, pval=P, equal_var=!H)
        end
    else
        if dv === nothing && group === nothing
            # Wide format
            numdata = data[ :, colwise(x -> (eltype(x) <: Number), data)]

            k = length(names(data))
            if k < 2
                throw(DomainError(data, "There should be at least 2 lists"))
            end

            samples = []
            for (i, feature) in enumerate(propertynames(numdata))
                insert!(samples, i, data[feature])
            end

            H, W, P = func(samples, α=α)
            if method == "levene"
                return DataFrame(W=W, pval=P, equal_var=!H)
            elseif method == "bartlett"
                return DataFrame(T=W, pval=P, equal_var=!H)
            end
        else
            # long format
            group = Symbol(group)
            dv = Symbol(dv)

            @assert group in propertynames(data)
            @assert dv in propertynames(data)

            grp = groupby(data, group, sort=false)
            
            samples = []
            for (i, subdf) in enumerate(grp)
                insert!(samples, i, DataFrame(subdf)[dv])
            end

            H, W, P = func(samples, α=α)
            if method == "levene"
                return DataFrame(W=W, pval=P, equal_var=!H)
            elseif method == "bartlett"
                return DataFrame(T=W, pval=P, equal_var=!H)
            end
        end
    end
end


"""
    sphericity(data[, dv, within, subject, method, α])

Mauchly and JNS test for sphericity.

Arguments
----------
- `data::DataFrame`: DataFrame containing the repeated measurements. Only long-format dataframe are supported for this function.
- `dv::Union{Nothing, String, Symbol}`: Name of column containing the dependent variable.
- `within::Union{Array{String}, Array{Symbol}, Nothing, String, Symbol}`: Name of column containing the within factor. If `within` is a list with two strings, this function computes the epsilon factor for the interaction between the two within-subject factor.
- `subject::Union{Nothing, String, Symbol}`: Name of column containing the subject identifier (only required if `data` is in long format).
- `method::String`: Method to compute sphericity:
    * `"jns"`: John, Nagao and Sugiura test.
    * `"mauchly"`: Mauchly test (default).
- `α::Float64`: Significance level

Returns
-------
- `spher::Bool`: True if data have the sphericity property.
- `W::Float64:` Test statistic.
- `chi2::Float64`: Chi-square statistic.
- `dof::Int64`:  Degrees of freedom.
- `pval::Flot64`: P-value.

Throwing
------
`DomainError` When testing for an interaction, if both within-subject factors have more than 2 levels (not yet supported in Pingouin).

See Also
--------
- [`epsilon`](@ref): Epsilon adjustement factor for repeated measures.
- [`homoscedasticity`](@ref): Test equality of variance.
- [`normality`](@ref): Univariate normality test.

Notes
-----
The **Mauchly** \$W\$ statistic [1] is defined by:

\$W = \\frac{\\prod \\lambda_j}{(\\frac{1}{k-1} \\sum \\lambda_j)^{k-1}}\$

where \$\\lambda_j\$ are the eigenvalues of the population
covariance matrix (= double-centered sample covariance matrix) and
\$k\$ is the number of conditions.

From then, the \$W\$ statistic is transformed into a chi-square
score using the number of observations per condition \$n\$

\$f = \\frac{2(k-1)^2+k+1}{6(k-1)(n-1)}\$
\$\\chi_w^2 = (f-1)(n-1) \\text{log}(W)\$

The p-value is then approximated using a chi-square distribution:

\$\\chi_w^2 \\sim \\chi^2(\\frac{k(k-1)}{2}-1)\$

The **JNS** \$V\$ statistic ([2], [3], [4]) is defined by:

\$V = \\frac{(\\sum_j^{k-1} \\lambda_j)^2}{\\sum_j^{k-1} \\lambda_j^2}\$

\$\\chi_v^2 = \\frac{n}{2}  (k-1)^2 (V - \\frac{1}{k-1})\$

and the p-value approximated using a chi-square distribution

\$\\chi_v^2 \\sim \\chi^2(\\frac{k(k-1)}{2}-1)\$

Missing values are automatically removed from `data` (listwise deletion).

References
----------
[1] Mauchly, J. W. (1940). Significance test for sphericity of a normal
n-variate distribution. The Annals of Mathematical Statistics,
11(2), 204-209.

[2] Nagao, H. (1973). On some test criteria for covariance matrix.
The Annals of Statistics, 700-709.

[3] Sugiura, N. (1972). Locally best invariant test for sphericity and
the limiting distributions. The Annals of Mathematical Statistics,
1312-1316.

[4] John, S. (1972). The distribution of a statistic used for testing
sphericity of normal distributions. Biometrika, 59(1), 169-173.

See also http://www.real-statistics.com/anova-repeated-measures/sphericity/

Examples
--------
Mauchly test for sphericity using a wide-format dataframe

```julia-repl
julia> data = DataFrame(A = [2.2, 3.1, 4.3, 4.1, 7.2],
                     B = [1.1, 2.5, 4.1, 5.2, 6.4],
                     C = [8.2, 4.5, 3.4, 6.2, 7.2])
julia> Pingouin.sphericity(data)
│ Row │ spher │ W        │ chi2    │ dof     │ pval      │
│     │ Bool  │ Float64  │ Float64 │ Float64 │ Float64   │
├─────┼───────┼──────────┼─────────┼─────────┼───────────┤
│ 1   │ 1     │ 0.210372 │ 4.67663 │ 2.0     │ 0.0964902 │
```

John, Nagao and Sugiura (JNS) test

```julia-repl
julia> Pingouin.sphericity(data, method="jns")[:pval]  # P-value only
0.045604240307520305
```

Now using a long-format dataframe

```julia-repl
julia> data = Pingouin.read_dataset("rm_anova2")
julia> head(data)
6x4 DataFrame
│ Row │ Subject │ Time   │ Metric  │ Performance │
│     │ Int64   │ String │ String  │ Int64       │
├─────┼─────────┼────────┼─────────┼─────────────┤
│ 1   │ 1       │ Pre    │ Product │ 13          │
│ 2   │ 2       │ Pre    │ Product │ 12          │
│ 3   │ 3       │ Pre    │ Product │ 17          │
│ 4   │ 4       │ Pre    │ Product │ 12          │
│ 5   │ 5       │ Pre    │ Product │ 19          │
│ 6   │ 6       │ Pre    │ Product │ 6           │
```

Let's first test sphericity for the *Time* within-subject factor

```julia-repl
julia> Pingouin.sphericity(data, dv=:Performance, subject=:Subject,
                        within=:Time)
│ Row │ spher │ W       │ chi2    │ dof   │ pval    │
│     │ Bool  │ Float64 │ Float64 │ Int64 │ Float64 │
├─────┼───────┼─────────┼─────────┼───────┼─────────┤
│ 1   │ 1     │ NaN     │ NaN     │ 1     │ 1.0     │
```

Since *Time* has only two levels (Pre and Post), the sphericity assumption
is necessarily met.

The *Metric* factor, however, has three levels:

```julia-repl
julia> Pingouin.sphericity(data, dv="Performance", subject="Subject",
                        within=["Metric"])[:pval]
0.8784417991645139
```

The p-value value is very large, and the test therefore indicates that
there is no violation of sphericity.

Now, let's calculate the epsilon for the interaction between the two
repeated measures factor. The current implementation in Pingouin only works
if at least one of the two within-subject factors has no more than two
levels.

```julia-repl
julia> Pingouin.sphericity(data, dv="Performance",
                        subject="Subject",
                        within=["Time", "Metric"])
│ Row │ spher │ W        │ chi2    │ dof     │ pval     │
│     │ Bool  │ Float64  │ Float64 │ Float64 │ Float64  │
├─────┼───────┼──────────┼─────────┼─────────┼──────────┤
│ 1   │ 1     │ 0.624799 │ 3.7626  │ 2.0     │ 0.152392 │
```

Here again, there is no violation of sphericity acccording to Mauchly's
test.
"""
function sphericity(data::DataFrame;
                    dv::Union{Nothing, String, Symbol}=nothing, 
                    within::Union{Array{String}, Array{Symbol}, Nothing, String, Symbol}=nothing, 
                    subject::Union{Nothing, String, Symbol}=nothing,
                    method::String="mauchly",
                    α=.05)::DataFrame
    levels = nothing
    if all([(v !== nothing) for v in [dv, within, subject]])
        data, levels = _transform_rm(data, dv=dv, within=within, subject=subject)

        if levels !== nothing
            if all(levels .> 2)
                throw(DomainError(levels, "If using two-way repeated measures design, at least one factor must have exactly two levels. More complex designs are not yet supported."))
            end
            if levels[1] < 2
                levels = nothing
            end
        end
    end

    # todo: dropna

    # todo: Support for two-way factor of shape (2, N)
    # data = _check_multilevel_rm(data, func='mauchly')

    # From here, we work only with one-way design
    n, k = size(data)
    d = k - 1

    # Sphericity is always met with only two repeated measures.
    if k <= 2
        return DataFrame(spher = true, W = NaN, chi2 = NaN, dof = 1, pval=1.)
    end

    # Compute dof of the test
    ddof = (d * (d + 1)) / 2 - 1
    ddof = ddof == 0 ? 1 : ddof

    if method == "mauchly"
        # Method 1. Contrast matrix. Similar to R & Matlab implementation.
        # Only works for one-way design or two-way design with shape (2, N).
        # 1 - Compute the successive difference matrix Z.
        #     (Note that the order of columns does not matter.)
        # 2 - Find the contrast matrix that M so that data * M = Z
        # 3 - Performs the QR decomposition of this matrix (= contrast matrix)
        # 4 - Compute sample covariance matrix S
        # 5 - Compute Mauchly's statistic
        # Z = data.diff(axis=1).dropna(axis=1)
        # M = np.linalg.lstsq(data, Z, rcond=None)[0]
        # C, _ = np.linalg.qr(M)
        # S = data.cov()
        # A = C.T.dot(S).dot(C)
        # logW = np.log(np.linalg.det(A)) - d * np.log(np.trace(A / d))
        # W = np.exp(logW)

        # Method 2. Eigenvalue-based method. Faster.
        # 1 - Estimate the population covariance (= double-centered)
        # 2 - Calculate n-1 eigenvalues
        # 3 - Compute Mauchly's statistic
        S = cov(convert(Matrix, data))
        S_pop = S .- mean(S, dims=2) .- mean(S, dims=1) .+ mean(S)
        eig = eigvals(S_pop)[2:end]
        eig = eig[eig .> 0.001]  # Additional check to remove very low eig
        W = prod(eig) / (sum(eig) / d)^d
        logW = log(W)

        # Compute chi-square and p-value (adapted from the ezANOVA R package)
        f = 1 - (2 * d^2 + d + 2) / (6 * d * (n - 1))
        w2 = ((d + 2) * (d - 1) * (d - 2) * (2 * d^3 + 6 * d^2 + 3 * k + 2) / (288 * ((n - 1) * d * f)^2))
        chi_sq = -(n - 1) * f * logW
        p1 = ccdf(Chisq(ddof), chi_sq)
        p2 = ccdf(Chisq(ddof + 4), chi_sq)

        pval = p1 + w2 * (p2 - p1)
    else
        # Method = JNS
        eps = epsilon(data, correction="gg")
        W = eps * d
        chi_sq = 0.5 * n * d^2 * (W - 1 / d)
        pval = ccdf(Chisq(ddof), chi_sq)
    end
    spher = pval > α ? true : false

    return DataFrame(spher = spher, W = W, chi2 = chi_sq, dof = ddof, pval=pval)
end


"""
    epsilon(data[, dv, within, subject, correction])

Epsilon adjustement factor for repeated measures.

Arguments
----------
- `data::Union{Array{<:Number}, DataFrame}`: DataFrame containing the repeated measurements. Only long-format dataframe are supported for this function.
- `dv::Union{Symbol, String, Nothing}`: Name of column containing the dependent variable.
- `within::Union{Array{String}, Array{Symbol}, Symbol, String, Nothing}`: Name of column containing the within factor (only required if `data` is in long format). If `within` is a list with two strings, this function computes the epsilon factor for the interaction between the two within-subject factor.
- `subject::Union{Symbol, String, Nothing}`: Name of column containing the subject identifier (only required if `data` is in long format).
- `correction::String`: Specify the epsilon version:
    * `"gg"`: Greenhouse-Geisser,
    * `"hf"`: Huynh-Feldt,
    * `"lb"`: Lower bound.

Returns
-------
eps::Float64: Epsilon adjustement factor.

See Also
--------
- [`sphericity`](@ref): Mauchly and JNS test for sphericity.
- [`homoscedasticity`](@ref): Test equality of variance.

Notes
-----
The lower bound epsilon is:

\$lb = \\frac{1}{\\text{dof}}\$,

where the degrees of freedom \$\\text{dof}\$ is the number of groups
\$k\$ minus 1 for one-way design and \$(k_1 - 1)(k_2 - 1)\$
for two-way design

The Greenhouse-Geisser epsilon is given by:

\$\\epsilon_{GG} = \\frac{k^2(\\overline{\\text{diag}(S)} - \\overline{S})^2}{(k-1)(\\sum_{i=1}^{k}\\sum_{j=1}^{k}s_{ij}^2 - 2k\\sum_{j=1}^{k}\\overline{s_i}^2 + k^2\\overline{S}^2)}\$

where \$S\$ is the covariance matrix, \$\\overline{S}\$ the
grandmean of S and \$\\overline{\\text{diag}(S)}\$ the mean of all the
elements on the diagonal of S (i.e. mean of the variances).

The Huynh-Feldt epsilon is given by:

\$\\epsilon_{HF} = \\frac{n(k-1)\\epsilon_{GG}-2}{(k-1) (n-1-(k-1)\\epsilon_{GG})}\$

where \$n\$ is the number of observations.

Missing values are automatically removed from data (listwise deletion).

Examples
--------
Using a wide-format dataframe

```julia-repl
julia> data = DataFrame(A = [2.2, 3.1, 4.3, 4.1, 7.2],
                     B = [1.1, 2.5, 4.1, 5.2, 6.4],
                     C = [8.2, 4.5, 3.4, 6.2, 7.2])
julia> Pingouin.epsilon(data, correction="gg")
0.5587754577585018
julia> Pingouin.epsilon(data, correction="hf")
0.6223448311539789
julia> Pingouin.epsilon(data, correction="lb")
0.5
```

Now using a long-format dataframe

```julia-repl
julia> data = Pingouin.read_dataset("rm_anova2")
julia> head(data)
6×4 DataFrame
│ Row │ Subject │ Time   │ Metric  │ Performance │
│     │ Int64   │ String │ String  │ Int64       │
├─────┼─────────┼────────┼─────────┼─────────────┤
│ 1   │ 1       │ Pre    │ Product │ 13          │
│ 2   │ 2       │ Pre    │ Product │ 12          │
│ 3   │ 3       │ Pre    │ Product │ 17          │
│ 4   │ 4       │ Pre    │ Product │ 12          │
│ 5   │ 5       │ Pre    │ Product │ 19          │
│ 6   │ 6       │ Pre    │ Product │ 6           │
```

Let's first calculate the epsilon of the *Time* within-subject factor

```julia-repl
julia> Pingouin.epsilon(data, dv="Performance", subject="Subject",
                     within="Time")
1.0
```

Since *Time* has only two levels (Pre and Post), the sphericity assumption
is necessarily met, and therefore the epsilon adjustement factor is 1.

The *Metric* factor, however, has three levels:

```julia-repl
julia> Pingouin.epsilon(data, dv=:Performance, subject=:Subject,
                     within=[:Metric])
0.9691029584899762
```

The epsilon value is very close to 1, meaning that there is no major
violation of sphericity.

Now, let's calculate the epsilon for the interaction between the two
repeated measures factor:

```julia-repl
julia> Pingouin.epsilon(data, dv=:Performance, subject=:Subject,
                     within=[:Time, :Metric])
0.727166420214127
```
"""
function epsilon(data::Union{Array{<:Number}, DataFrame};
                 dv::Union{Symbol, String, Nothing}=nothing,
                 within::Union{Array{String}, Array{Symbol}, Symbol, String, Nothing}=nothing,
                 subject::Union{Symbol, String, Nothing}=nothing,
                 correction::String="gg")::Float64
    levels = nothing
    if all([(v !== nothing) for v in [dv, within, subject]])
        data, levels = _transform_rm(data, dv=dv, within=within, subject=subject)

        if levels !== nothing
            if all(levels .> 2)
                @warn "Epsilon values might be innaccurate in two-way repeated measures design where each factor has more than 2 levels.\nPlease double-check your results."
            end
            if levels[1] < 2
                levels = nothing
            end
        end
    end
    
    # todo: drop na

    S = cov(convert(Matrix, data))
    n, k = size(data)

    # Epsilon is always 1 with only two repeated measures.
    if (k <= 2) && levels === nothing
        return 1.
    end

    # degrees of freedom
    # one-way and two-way designs
    df = levels === nothing ? 
                            k - 1. :
                            (levels[1] - 1) * (levels[2] - 1)


    if correction == "lb"
        return 1. / df
    end

    # Greenhouse-Geisser
    # Method 1. Sums of squares. (see real-statistics.com)
    mean_var = mean(diag(S))
    S_mean = mean(S)
    ss_mat = sum(S.^2)
    ss_rows = sum(mean(S, dims=2).^2)
    num = (k * (mean_var - S_mean)) ^ 2
    den = (k - 1) * (ss_mat - 2 * k * ss_rows + k^2 * S_mean^2)
    eps = minimum([num / den, 1.])

    # Huynh-Feldt
    if correction == "hf"
        num = n * df * eps - 2
        den = df * (n - 1 - df * eps)
        eps = minimum([num / den, 1.])
    end

    return eps
end


function _transform_rm(data::DataFrame;
                       dv::Union{Symbol, String, Nothing}=nothing,
                       within::Union{Symbol, String, Nothing, Array{String}, Array{Symbol}}=nothing,
                       subject::Union{Symbol, String, Nothing}=nothing)::Tuple{Matrix, Array{Int64}}
    """
    Convert long-format dataframe (one and two-way designs).
    This internal function is used in Pingouin.epsilon and Pingouin.sphericity.
    """
    @assert Symbol(dv) in propertynames(data)
    @assert Symbol(subject) in propertynames(data)
    @assert !any(isnan.(data[dv]))
    if isa(within, Union{String, Symbol})
        within = [within]
    end
    for w in within
        @assert Symbol(w) in propertynames(data)
    end

    data = data[[subject, within..., dv]]
    grp = combine(groupby(data, [subject, within...], skipmissing=true), dv => mean => dv)
    if length(within) == 1
        grp = unstack(grp, subject, within..., dv)
        grp = grp[:, Not(subject)]

        return convert(Matrix{Real}, grp), nothing
    elseif length(within) == 2
        # data_wide = unstack(data, within[end], dv)
        # combine(groupby(data_wide, [subject]), sort(unique(data[within[end]])) .=> diff .=> sort(unique(data[within[end]])))
        function _factor_levels(data::DataFrame, within::Array)::Array
            levels = []
            for factor in within
                l = length(unique(data[factor]))
                push!(levels, l)
            end
    
            return levels
        end

        within_levels = _factor_levels(data, within)
        within = within[sortperm(within_levels)]
        sort!(within_levels)

        @assert within_levels[2] >= 2 "Factor must have at least two levels"

        grp = groupby(grp, within[1], skipmissing=true)
        grp = [g[:, Not(within[1])] for g in grp]
        grp = unstack.(grp, subject, within[2], dv)
        grp = [g[:, Not(subject)] for g in grp]
    
        mat = grp[1]
        for i in 2:length(grp)
            mat .-= grp[i]
        end

        return convert(Matrix{Real}, mat), within_levels
    else
        throw(DomainError(within, "Only one-way and two-way designs are supported."))
    end
end
