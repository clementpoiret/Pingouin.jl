using Distributions
using Roots

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
        return function _two_sided(r::Float64,
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
        return function _greater(r::Float64,
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
        return function _less(r::Float64,
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
