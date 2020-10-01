using DataFrames

include("shapiro.jl")

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

2. Long-format dataframe

>>> data = ...
>>> pg.normality(data, dv='Performance', group='Time')
             W      pval  normal
Pre   0.967718  0.478773    True
Post  0.940728  0.095157    True
"""
function normality(data, dv=nothing, group=nothing, method::String="shapiro", α::Float64=0.05)
    # todo: handle series and arrays
    # todo: handle long format

    # print("normality test for data, on $group using $method, at p=$alpha")
    func = eval(Meta.parse(method))
    if isa(data, Array{})
        return func(data, α)
    end
end

function shapiro(x::Array{}, α::Float64=0.05)::DataFrame
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
