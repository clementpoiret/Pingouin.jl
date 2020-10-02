using DataFrames
using HypothesisTests

include("_shapiro.jl")

"""
Univariate normality test.

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
    Supported values: ["shapiro", "jarque_bera"].
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

The Jarque-Bera statistic is to test the null hypothesis that a real-valued vector `y`
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
>>> dataset = DataFrame(CSV.File("Pingouin/datasets/anova.csv"))
>>> Pingouin.normality(dataset["Pain Threshold"])
1×3 DataFrame
│ Row │ W         │ pval     │ normal │
│     │ Float64   │ Float64  │ Bool   │
├─────┼───────────┼──────────┼────────┤
│ 1   │ -0.842541 │ 0.800257 │ 1      │

2. Wide-format dataframe using Jarque-Bera test

>>> dataset = DataFrame(CSV.File("Pingouin/datasets/mediation.csv"))
>>> Pingouin.normality(dataset, method="jarque_bera")
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

"""
Compute the Shapiro-Wilk statistic to test the null hypothesis that a real-valued vector `y` is normally distributed.
"""
function shapiro(x::Array{}, α::Float64=0.05)::DataFrame
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

"""
Compute the Jarque-Bera statistic to test the null hypothesis that a real-valued vector `y` is normally distributed.
"""
function jarque_bera(x::Array{}, α::Float64=0.05)::DataFrame
    test = JarqueBeraTest(x)

    JB = test.JB
    P = pvalue(test)
    H = (α >= P)

    return DataFrame(W=JB, pval=P, normal=!H)
end
