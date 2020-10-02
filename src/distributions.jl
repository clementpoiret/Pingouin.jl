using DataFrames

include("_shapiro.jl")

"""Univariate normality test.

Parameters
----------
data : `DataFrame`, series, list or 1D array
    Iterable. Can be either a single list, 1D array,
    or a wide- or long-format dataframe.
dv : str
    Dependent variable (only when ``data`` is a long-format dataframe).
group : str
    Grouping variable (only when ``data`` is a long-format dataframe).
method : str
    Normality test. `'shapiro'` (default) performs the Shapiro-Wilk test
    using the AS R94 algorithm. If the kurtosis is higher than 3, it 
    performs a Shapiro-Francia test for leptokurtic distributions.
alpha : float64
    Significance level.

Returns
-------
stats : `DataFrame`

    * ``'W'``: Test statistic.
    * ``'pval'``: p-value.
    * ``'normal'``: True if ``data`` is normally distributed.

See Also
--------
homoscedasticity : Test equality of variance.
sphericity : Mauchly's test for sphericity.

Notes
-----
The Shapiro-Wilk test calculates a :math:`W` statistic that tests whether a
random sample :math:`x_1, x_2, ..., x_n` comes from a normal distribution.

The :math:`W` is normalized (:math:`W = (W - μ) / σ`)

The null-hypothesis of this test is that the population is normally
distributed. Thus, if the p-value is less than the
chosen alpha level (typically set at 0.05), then the null hypothesis is
rejected and there is evidence that the data tested are not normally
distributed.

The result of the Shapiro-Wilk test should be interpreted with caution in
the case of large sample sizes (>5000). Indeed, quoting from
`Wikipedia <https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test>`_:

    *"Like most statistical significance tests, if the sample size is
    sufficiently large this test may detect even trivial departures from
    the null hypothesis (i.e., although there may be some statistically
    significant effect, it may be too small to be of any practical
    significance); thus, additional investigation of the effect size is
    typically advisable, e.g., a Q–Q plot in this case."*

Note that missing values are automatically removed (casewise deletion).

References
----------
* Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test
  for normality (complete samples). Biometrika, 52(3/4), 591-611.

* https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

Examples
--------
1. Shapiro-Wilk test on a 1D array
>>> dataset = DataFrame(CSV.File("Pingouin/datasets/anova.csv"))
>>> Pingouin.normality(dataset["Pain Threshold"])
1×3 DataFrame
│ Row │ W         │ pval     │ normal │
│     │ Float64   │ Float64  │ Bool   │
├─────┼───────────┼──────────┼────────┤
│ 1   │ -0.842541 │ 0.800257 │ 1      │

2. Wide-format dataframe

>>> dataset = DataFrame(CSV.File("Pingouin/datasets/mediation.csv"))
>>> Pingouin.normality(dataset)
│ Row │ dv     │ W       │ pval        │ normal │
│     │ Symbol │ Float64 │ Float64     │ Bool   │
├─────┼────────┼─────────┼─────────────┼────────┤
│ 1   │ X      │ 2.80237 │ 0.00253646  │ 0      │
│ 2   │ M      │ 1.85236 │ 0.0319872   │ 0      │
│ 3   │ Y      │ 2.00508 │ 0.0224772   │ 0      │
│ 4   │ Mbin   │ 7.56875 │ 1.88738e-14 │ 0      │
│ 5   │ Ybin   │ 7.55459 │ 2.09832e-14 │ 0      │
│ 6   │ W1     │ 2.73591 │ 0.00311037  │ 0      │
│ 7   │ W2     │ 8.22676 │ 1.11022e-16 │ 0      │

3. Long-format dataframe
>>> dataset = DataFrame(CSV.File("Pingouin/datasets/rm_anova2.csv"))
>>> Pingouin.normality(dataset, dv=:Performance, group=:Time)
│ Row │ Time   │ W         │ pval      │ normal │
│     │ String │ Float64   │ Float64   │ Bool   │
├─────┼────────┼───────────┼───────────┼────────┤
│ 1   │ Pre    │ 0.0532374 │ 0.478771  │ 1      │
│ 2   │ Post   │ 1.30965   │ 0.0951576 │ 1      │

"""
function normality(data; dv=nothing, group=nothing, method::String="shapiro", α::Float64=0.05)
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
    
function shapiro(x::Array{}, α::Float64=0.05)::DataFrame
    x = x[@. !isnan.(x)]
    
    n = length(x)

    if n <= 3
        throw(DomainError(x, "Data must be at least length 4."))
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
