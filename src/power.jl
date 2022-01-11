using Distributions
using Roots

"""
    power_ttest(d, n, power, α[, contrast, tail])

Evaluate power, sample size, effect size or significance level of a one-sample T-test,
a paired T-test or an independent two-samples T-test with equal sample sizes.

Arguments
---------
- `d::Float64`: Cohen d effect size
- `n::Int64`: Sample size. In case of a two-sample T-test, sample sizes are assumed to be equal. Otherwise, see the [`power_ttest2n`](@ref) function.
- `power::Float64`: Test power (= 1 - type II error).
- `α::Float64`: Significance level (type I error probability). The default is 0.05.
- `contrast::String`: Can be `"one-sample"`, `"two-samples"` or `"paired"`. Note that `"one-sample"` and `"paired"` have the same behavior.
- `tail::Symbol`: Defines the alternative hypothesis, or tail of the test. Must be one of :both (default), :right or :left.

Notes
-----
Exactly ONE of the parameters `d`, `n`, `power` and `α` must
be passed as Nothing, and that parameter is determined from the others.

For a paired T-test, the sample size `n` corresponds to the number of
pairs. For an independent two-sample T-test with equal sample sizes, `n`
corresponds to the sample size of each group (i.e. number of observations
in one group). If the sample sizes are unequal, please use the
[`power_ttest2n`](@ref) function instead.

Notice that `α` has a default value of 0.05 so Nothing must be
explicitly passed if you want to compute it.

This function is a Python adaptation of the `pwr.t.test`
function implemented in the
`pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>` R package.

Statistical power is the likelihood that a study will
detect an effect when there is an effect there to be detected.
A high statistical power means that there is a low probability of
concluding that there is no effect when there is one.
Statistical power is mainly affected by the effect size and the sample
size.

The first step is to use the Cohen's d to calculate the non-centrality
parameter ``\\delta`` and degrees of freedom ``v``.
In case of paired groups, this is:

\$\\delta = d * \\sqrt n\$
\$v = n - 1\$

and in case of independent groups with equal sample sizes:

\$\\delta = d * \\sqrt{\\frac{n}{2}}\$
\$v = (n - 1) * 2\$

where ``d`` is the Cohen d and ``n`` the sample size.

The critical value is then found using the percent point function of the T
distribution with ``q = 1 - α`` and ``v``
degrees of freedom.

Finally, the power of the test is given by the survival function of the
non-central distribution using the previously calculated critical value,
degrees of freedom and non-centrality parameter.

`brenth` is used to solve power equations for other
variables (i.e. sample size, effect size, or significance level). If the
solving fails, a nan value is returned.

Results have been tested against GPower and the
`pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>` R package.

Examples
--------
1. Compute power of a one-sample T-test given ``d``, ``n`` and ``\\alpha``

```julia-repl
julia> using Pingouin
julia> Pingouin.power_ttest(0.5, 20, nothing, 0.05, contrast="one-sample")
0.5645044184390837
```

2. Compute required sample size given ``d``, ``power`` and ``\\alpha``

```julia-repl
julia> Pingouin.power_ttest(0.5, nothing, 0.80, 0.05, tail=:right)
50.15078338685538
```

3. Compute achieved ``d`` given ``n``, ``power`` and ``\\alpha`` level

```julia-repl
julia> Pingouin.power_ttest(nothing, 20, 0.80, 0.05, contrast="paired")
0.6604416546228311
```

4. Compute achieved alpha level given ``d``, ``n`` and ``power``

```julia-repl
julia> Pingouin.power_ttest(0.5, 20, 0.80, nothing)
0.4430167658448699
```

5. One-sided tests

```julia-repl
julia> Pingouin.power_ttest(0.5, 20, nothing, 0.05, tail=:right)
0.4633743492964484
```

```julia-repl
julia> Pingouin.power_ttest(0.5, 20, nothing, 0.05, tail=:left)
0.0006909466675597553
```
"""
function _get_f_power_ttest(tail::Symbol,
    tsample::Int64,
    tside::Int64)::Function

    if tail == :both
        return function _two_sided_ttest(d::Float64,
            n::Real,
            α::Float64)::Float64

            ddof = (n - 1) * tsample
            nc = d * sqrt(n / tsample)
            tcrit = quantile(TDist(ddof), 1 - α / tside)

            return ccdf(NoncentralT(ddof, nc), tcrit) + cdf(NoncentralT(ddof, nc), -tcrit)
        end
    elseif tail == :right
        return function _greater_ttest(d::Float64,
            n::Real,
            α::Float64)::Float64

            ddof = (n - 1) * tsample
            nc = d * sqrt(n / tsample)
            tcrit = quantile(TDist(ddof), 1 - α / tside)

            return ccdf(NoncentralT(ddof, nc), tcrit)
        end
    elseif tail == :left
        return function _less_ttest(d::Float64,
            n::Real,
            α::Float64)::Float64

            ddof = (n - 1) * tsample
            nc = d * sqrt(n / tsample)
            tcrit = quantile(TDist(ddof), α / tside)
            return cdf(NoncentralT(ddof, nc), tcrit)
        end
    end
end
# Finds α
function power_ttest(d::Float64,
    n::Int64,
    power::Float64,
    α::Nothing;
    contrast::String = "two-samples",
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    @assert contrast in ["one-sample", "two-samples", "paired"] "Contrast must be one of 'one-sample', 'two-samples' or 'paired'."
    tsample = contrast == "two-samples" ? 2 : 1
    tside = tail == :both ? 2 : 1
    if tside == 2
        d = abs(d)
    end
    @assert 0 < power <= 1

    # Compute achieved α (significance) level given d, n and power
    _find_α(α) = _get_f_power_ttest(tail, tsample, tside)(d, n, α) - power

    return fzero(_find_α, (1e-10, 1 - 1e-10), Roots.Brent())
end
# Finds power
function power_ttest(d::Float64,
    n::Int64,
    power::Nothing,
    α::Float64;
    contrast::String = "two-samples",
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    @assert contrast in ["one-sample", "two-samples", "paired"] "Contrast must be one of 'one-sample', 'two-samples' or 'paired'."
    tsample = contrast == "two-samples" ? 2 : 1
    tside = tail == :both ? 2 : 1
    if tside == 2
        d = abs(d)
    end
    @assert 0 < α <= 1

    # Compute achieved power given d, n and α
    return _get_f_power_ttest(tail, tsample, tside)(d, n, α)
end
# Finds n
function power_ttest(d::Float64,
    n::Nothing,
    power::Float64,
    α::Float64;
    contrast::String = "two-samples",
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    @assert contrast in ["one-sample", "two-samples", "paired"] "Contrast must be one of 'one-sample', 'two-samples' or 'paired'."
    tsample = contrast == "two-samples" ? 2 : 1
    tside = tail == :both ? 2 : 1
    if tside == 2
        d = abs(d)
    end
    @assert 0 < α <= 1
    @assert 0 < power <= 1

    # Compute required sample size given d, power and α
    _find_n(n) = _get_f_power_ttest(tail, tsample, tside)(d, n, α) - power

    return fzero(_find_n, (2 + 1e-10, 1e+07), Roots.Brent())
end
# Finds d
function power_ttest(d::Nothing,
    n::Int64,
    power::Float64,
    α::Float64;
    contrast::String = "two-samples",
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    @assert contrast in ["one-sample", "two-samples", "paired"] "Contrast must be one of 'one-sample', 'two-samples' or 'paired'."
    tsample = contrast == "two-samples" ? 2 : 1
    tside = tail == :both ? 2 : 1
    @assert 0 < α <= 1
    @assert 0 < power <= 1

    # Compute achieved d given sample size, power and α level
    _find_d(d) = _get_f_power_ttest(tail, tsample, tside)(d, n, α) - power

    if tail == :both
        b0, b1 = 1e-07, 10
    elseif tail == :left
        b0, b1 = -10, 5
    elseif tail == :right
        b0, b1 = -5, 10
    end

    return fzero(_find_d, (b0, b1), Roots.Brent())
end


"""
    power_ttest2n(nx, ny, d, power, α[, contrast, tail])

Evaluate power, effect size or  significance level of an independent two-samples T-test with unequal sample sizes.

Arguments
---------
- `nx, ny::Int64`: Sample sizes, must be specified. If the sample sizes are equal, you should use the [`power_ttest`](@ref) function.
- `d::Float64`: Cohen d effect size
- `power::Float64`: Test power (= 1 - type II error).
- `α::Float64`: Significance level (type I error probability). The default is 0.05.
- `contrast::String`: Can be `"one-sample"`, `"two-samples"` or `"paired"`. Note that `"one-sample"` and `"paired"` have the same behavior.
- `tail::Symbol`: Defines the alternative hypothesis, or tail of the test. Must be one of :both (default), :right or :left.

Notes
-----
Exactly ONE of the parameters `d`, `power` and `α` must
be passed as Nothing, and that parameter is determined from the others.

Notice that `α` has a default value of 0.05 so Nothing must be
explicitly passed if you want to compute it.

This function is a Python adaptation of the `pwr.t2n.test`
function implemented in the
`pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>` R package.

Statistical power is the likelihood that a study will
detect an effect when there is an effect there to be detected.
A high statistical power means that there is a low probability of
concluding that there is no effect when there is one.
Statistical power is mainly affected by the effect size and the sample
size.

The first step is to use the Cohen's d to calculate the non-centrality
parameter ``\\delta`` and degrees of freedom ``v``.
In case of paired groups, this is:

\$\\delta = d * \\sqrt n\$
\$v = n - 1\$

and in case of independent groups with equal sample sizes:

\$\\delta = d * \\sqrt{\\frac{n}{2}}\$
\$v = (n - 1) * 2\$

where ``d`` is the Cohen d and ``n`` the sample size.

The critical value is then found using the percent point function of the T
distribution with ``q = 1 - α`` and ``v``
degrees of freedom.

Finally, the power of the test is given by the survival function of the
non-central distribution using the previously calculated critical value,
degrees of freedom and non-centrality parameter.

`brenth` is used to solve power equations for other
variables (i.e. sample size, effect size, or significance level). If the
solving fails, a nan value is returned.

Results have been tested against GPower and the
`pwr <https://cran.r-project.org/web/packages/pwr/pwr.pdf>` R package.

Examples
--------
1. Compute achieved power of a T-test given ``d``, ``n`` and ``\\alpha``

```julia-repl
julia> using Pingouin
julia> Pingouin.power_ttest2n(20, 15, 0.5, nothing, 0.05, tail=:right)
0.41641558972125337
```

2. Compute achieved ``d`` given ``n``, ``power`` and ``\\alpha`` level

```julia-repl
julia> Pingouin.power_ttest2n(20, 15, nothing, 0.80, 0.05)
0.9859223315621992
```

3. Compute achieved alpha level given ``d``, ``n`` and ``power``

```julia-repl
julia> Pingouin.power_ttest2n(20, 15, 0.5, 0.80, nothing)
0.4999843253701041
```
"""
function _get_f_power_ttest2n(tail::Symbol,
    tside::Int64)::Function

    if tail == :both
        return function _two_sided_ttest2n(d::Float64,
            nx::Real,
            ny::Real,
            α::Float64)::Float64

            ddof = nx + ny - 2
            nc = d * (1 / sqrt(1 / nx + 1 / ny))
            tcrit = quantile(TDist(ddof), 1 - α / tside)

            return ccdf(NoncentralT(ddof, nc), tcrit) + cdf(NoncentralT(ddof, nc), -tcrit)
        end
    elseif tail == :right
        return function _greater_ttest2n(d::Float64,
            nx::Real,
            ny::Real,
            α::Float64)::Float64

            ddof = nx + ny - 2
            nc = d * (1 / sqrt(1 / nx + 1 / ny))
            tcrit = quantile(TDist(ddof), 1 - α / tside)

            return ccdf(NoncentralT(ddof, nc), tcrit)
        end
    elseif tail == :left
        return function _less_ttest2n(d::Float64,
            nx::Real,
            ny::Real,
            α::Float64)::Float64

            ddof = nx + ny - 2
            nc = d * (1 / sqrt(1 / nx + 1 / ny))
            tcrit = quantile(TDist(ddof), α / tside)
            return cdf(NoncentralT(ddof, nc), tcrit)
        end
    end
end
# Finds α
function power_ttest2n(nx::Int64,
    ny::Int64,
    d::Float64,
    power::Float64,
    α::Nothing;
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    tside = tail == :both ? 2 : 1
    if tside == 2
        d = abs(d)
    end
    @assert 0 < power <= 1

    # Compute achieved α (significance) level given d, n and power
    _find_α(α) = _get_f_power_ttest2n(tail, tside)(d, nx, ny, α) - power

    return fzero(_find_α, (1e-10, 1 - 1e-10), Roots.Brent())
end
# Finds power
function power_ttest2n(nx::Int64,
    ny::Int64,
    d::Float64,
    power::Nothing,
    α::Float64;
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    tside = tail == :both ? 2 : 1
    if tside == 2
        d = abs(d)
    end
    @assert 0 < α <= 1

    # Compute achieved power given d, n and α
    return _get_f_power_ttest2n(tail, tside)(d, nx, ny, α)
end
# Finds d
function power_ttest2n(nx::Int64,
    ny::Int64,
    d::Nothing,
    power::Float64,
    α::Float64;
    tail::Symbol = :both)::Float64

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."
    tside = tail == :both ? 2 : 1
    @assert 0 < α <= 1
    @assert 0 < power <= 1

    # Compute achieved d given sample size, power and α level
    _find_d(d) = _get_f_power_ttest2n(tail, tside)(d, nx, ny, α) - power

    if tail == :both
        b0, b1 = 1e-07, 10
    elseif tail == :left
        b0, b1 = -10, 5
    elseif tail == :right
        b0, b1 = -5, 10
    end

    return fzero(_find_d, (b0, b1), Roots.Brent())
end


"""
    power_corr(r, n, power, α[, alternative])

Evaluate power, sample size, correlation coefficient or
significance level of a correlation test.

Arguments
---------
- `r::Float64`: Correlation coefficient.
- `n::Real`: Sample size.
- `power::Float64`: Test power (= 1 - type II error).
- `α::Float64`: Significance level (type I error probability). Defaults to 0.05.
- `alternative::String`: Defines the alternative hypothesis, or tail of the correlation. Must be one of "two-sided" (default), "greater" or "less". Both "greater" and "less" return a one-sided p-value. "greater" tests against the alternative hypothesis that the correlation is positive (greater than zero), "less" tests against the hypothesis that the correlation is negative.

Notes
-----
Exactly ONE of the parameters `r`, `n`, `power` and `α` must
be passed as `nothing`, and that parameter is determined from the others.

Notice that `α` has a default value of 0.05 so `nothing` must be
explicitly passed if you want to compute it.

`brent` (from `Roots.jl``) is used to solve power equations for other
variables (i.e. sample size, effect size, or significance level). If the
solving fails, a NaN value is returned.

This function is an adaptation of the `pwr.r.test`
function implemented in the https://cran.r-project.org/web/packages/pwr/pwr.pdf R package.

Examples
--------

1. Compute achieved power given `r`, `n` and `α`

```julia-repl
julia> using Pingouin
julia> power = Pingouin.power_corr(0.5, 20, nothing, 0.05);
julia> power
0.6378746487367584
```

2. Same but one-sided test

```julia-repl
julia> Pingouin.power_corr(0.5, 20, nothing, 0.05, alternative="greater")
0.7509872964018015
julia> Pingouin.power_corr(0.5, 20, nothing, 0.05, alternative="less")
3.738124767341007e-5
```

3. Compute required sample size given `r`, `power` and `α`

```julia-repl
julia> n = Pingouin.power_corr(0.5, nothing, 0.6, 0.05);
julia> n
18.51787901242335
```

4. Compute achieved `r` given `n`, `power` and `α` level

```julia-repl
julia> r = Pingouin.power_corr(nothing, 20, 0.6, 0.05);
julia> r
0.48205541708037897
```

5. Compute achieved alpha level given `r`, `n` and `power`

```julia-repl
julia> α = Pingouin.power_corr(0.5, 20, 0.8, nothing)
0.13774641452888606
```
"""
function _get_f_power_corr(alternative::String)::Function

    if alternative == "two-sided"
        return function _two_sided_corr(r::Float64,
            n::Real,
            α::Float64)::Float64

            ddof = n - 2
            ttt = quantile(TDist(ddof), 1 - α / 2)
            rc = sqrt(ttt^2 / (ttt^2 + ddof))
            zr = atanh(r) + r / (2 * (n - 1))
            zrc = atanh(rc)
            power = cdf(Normal(), (zr - zrc) * sqrt(n - 3)) + cdf(Normal(), (-zr - zrc) * sqrt(n - 3))

            return power
        end
    elseif alternative == "greater"
        return function _greater_corr(r::Float64,
            n::Real,
            α::Float64)::Float64

            ddof = n - 2
            ttt = quantile(TDist(ddof), 1 - α)
            rc = sqrt(ttt^2 / (ttt^2 + ddof))
            zr = atanh(r) + r / (2 * (n - 1))
            zrc = atanh(rc)
            power = cdf(Normal(), (zr - zrc) * sqrt(n - 3))

            return power
        end
    elseif alternative == "less"
        return function _less_corr(r::Float64,
            n::Real,
            α::Float64)::Float64

            r = -r
            ddof = n - 2
            ttt = quantile(TDist(ddof), 1 - α)
            rc = sqrt(ttt^2 / (ttt^2 + ddof))
            zr = atanh(r) + r / (2 * (n - 1))
            zrc = atanh(rc)
            power = cdf(Normal(), (zr - zrc) * sqrt(n - 3))

            return power
        end
    end
end
# Finds α
function power_corr(r::Float64,
    n::Real,
    power::Float64,
    α::Nothing;
    alternative::String = "two-sided")::Float64

    @assert alternative in ["two-sided", "greater", "less"] "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    @assert -1 <= r <= 1 "Correlation coefficient must be between -1 and 1."
    if alternative == "two-sided"
        r = abs(r)
    end
    @assert 0 < power <= 1 "Power must be between 0 and 1."
    if n <= 4
        @warn "Sample size is too small to compute power."
        return NaN
    end

    # Compute achieved alpha (significance) level given r, n and power
    _find_α(α) = _get_f_power_corr(alternative)(r, n, α) - power

    return fzero(_find_α, (1e-10, 1 - 1e-10), Roots.Brent())
end
# Finds power
function power_corr(r::Float64,
    n::Real,
    power::Nothing,
    α::Float64 = 0.05;
    alternative::String = "two-sided")::Float64

    @assert alternative in ["two-sided", "greater", "less"] "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    @assert -1 <= r <= 1 "Correlation coefficient must be between -1 and 1."
    if alternative == "two-sided"
        r = abs(r)
    end
    @assert 0 < α <= 1 "Significance level must be between 0 and 1."
    if n <= 4
        @warn "Sample size is too small to compute power."
        return NaN
    end

    # Compute achieved power given r, n and alpha
    return _get_f_power_corr(alternative)(r, n, α)
end
# Finds n
function power_corr(r::Float64,
    n::Nothing,
    power::Float64,
    α::Float64 = 0.05;
    alternative::String = "two-sided")::Float64

    @assert alternative in ["two-sided", "greater", "less"] "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    @assert -1 <= r <= 1 "Correlation coefficient must be between -1 and 1."
    if alternative == "two-sided"
        r = abs(r)
    end
    @assert 0 < α <= 1 "Significance level must be between 0 and 1."
    @assert 0 < power <= 1 "Power must be between 0 and 1."

    # Compute required sample size given r, power and alpha
    _find_n(n) = _get_f_power_corr(alternative)(r, n, α) - power

    return fzero(_find_n, (4 + 1e-10, 1e+09), Roots.Brent())
end
# Finds r
function power_corr(r::Nothing,
    n::Real,
    power::Float64,
    α::Float64 = 0.05;
    alternative::String = "two-sided")::Float64

    @assert alternative in ["two-sided", "greater", "less"] "Alternative must be one of 'two-sided' (default), 'greater' or 'less'."
    @assert 0 < α <= 1 "Significance level must be between 0 and 1."
    @assert 0 < power <= 1 "Power must be between 0 and 1."
    if n <= 4
        @warn "Sample size is too small to compute power."
        return NaN
    end

    # Compute achieved r given sample size, power and α level
    _find_r(r) = _get_f_power_corr(alternative)(r, n, α) - power

    if alternative == "two-sided"
        return fzero(_find_r, (1e-10, 1 - 1e-10), Roots.Brent())
    else
        return fzero(_find_r, (-1 + 1e-10, 1 - 1e-10), Roots.Brent())
    end
end
