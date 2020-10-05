using DataFrames
using HypothesisTests

include("_shapiro.jl")
include("_homoscedasticity.jl")

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
    >>> dataset = Pingouin.read_dataset("anova")
    >>> Pingouin.normality(dataset["Pain Threshold"])
    1×3 DataFrame
    │ Row │ W         │ pval     │ normal │
    │     │ Float64   │ Float64  │ Bool   │
    ├─────┼───────────┼──────────┼────────┤
    │ 1   │ -0.842541 │ 0.800257 │ 1      │

    2. Wide-format dataframe using Jarque-Bera test

    >>> dataset = Pingouin.read_dataset("mediation")
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

    >>> dataset = Pingouin.read_dataset("rm_anova2")
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

"""
Test equality of variance.

    Parameters
    ----------
    data : `DataFrame` or array
        Iterable. Can be either an Array iterables or a wide- or long-format
        pandas dataframe.
    dv : str
        Dependent variable (only when ``data`` is a long-format dataframe).
    group : str
        Grouping variable (only when ``data`` is a long-format dataframe).
    method : str
        Statistical test. `'levene'` (default) performs the Levene test
        and `'bartlett'` performs the Bartlett test.
        The former is more robust to departure from normality.
    alpha : float
        Significance level.

    Returns
    -------
    stats : `DataFrame`

        * ``'W/T'``: Test statistic ('W' for Levene, 'T' for Bartlett)
        * ``'pval'``: p-value
        * ``'equal_var'``: True if ``data`` has equal variance

    See Also
    --------
    normality : Univariate normality test.
    sphericity : Mauchly's test for sphericity.

    Notes
    -----
    The **Bartlett** :math:`T` statistic [1]_ is defined as:

    .. math::

        T = \\frac{(N-k) \\ln{s^{2}_{p}} - \\sum_{i=1}^{k}(N_{i} - 1)
        \\ln{s^{2}_{i}}}{1 + (1/(3(k-1)))((\\sum_{i=1}^{k}{1/(N_{i} - 1))}
        - 1/(N-k))}

    where :math:`s_i^2` is the variance of the :math:`i^{th}` group,
    :math:`N` is the total sample size, :math:`N_i` is the sample size of the
    :math:`i^{th}` group, :math:`k` is the number of groups,
    and :math:`s_p^2` is the pooled variance.

    The pooled variance is a weighted average of the group variances and is
    defined as:

    .. math:: s^{2}_{p} = \\sum_{i=1}^{k}(N_{i} - 1)s^{2}_{i}/(N-k)

    The p-value is then computed using a chi-square distribution:

    .. math:: T \\sim \\chi^2(k-1)

    The **Levene** :math:`W` statistic [2]_ is defined as:

    .. math::

        W = \\frac{(N-k)} {(k-1)}
        \\frac{\\sum_{i=1}^{k}N_{i}(\\overline{Z}_{i.}-\\overline{Z})^{2} }
        {\\sum_{i=1}^{k}\\sum_{j=1}^{N_i}(Z_{ij}-\\overline{Z}_{i.})^{2} }

    where :math:`Z_{ij} = |Y_{ij} - \\text{median}({Y}_{i.})|`,
    :math:`\\overline{Z}_{i.}` are the group means of :math:`Z_{ij}` and
    :math:`\\overline{Z}` is the grand mean of :math:`Z_{ij}`.

    The p-value is then computed using a F-distribution:

    .. math:: W \\sim F(k-1, N-k)

    .. warning:: Missing values are not supported for this function.
        Make sure to remove them before using the
        :py:meth:`pandas.DataFrame.dropna` or :py:func:`pingouin.remove_na`
        functions.

    References
    ----------
    .. [1] Bartlett, M. S. (1937). Properties of sufficiency and statistical
           tests. Proc. R. Soc. Lond. A, 160(901), 268-282.

    .. [2] Brown, M. B., & Forsythe, A. B. (1974). Robust tests for the
           equality of variances. Journal of the American Statistical
           Association, 69(346), 364-367.

    Examples
    --------
    1. Levene test on a wide-format dataframe

    >>> data = Pingouin.read_dataset("mediation")
    >>> Pingouin.homoscedasticity(data[["X", "Y", "M"]])
    1×3 DataFrame
    │ Row │ W       │ pval     │ equal_var │
    │     │ Float64 │ Float64  │ Bool      │
    ├─────┼─────────┼──────────┼───────────┤
    │ 1   │ 1.17352 │ 0.310707 │ 1         │

    2. Bartlett test using an array of arrays

    >>> data = [[4, 8, 9, 20, 14], [5, 8, 15, 45, 12]]
    >>> Pingouin.homoscedasticity(data, method="bartlett", α=.05)
    1×3 DataFrame
    │ Row │ T       │ pval     │ equal_var │
    │     │ Float64 │ Float64  │ Bool      │
    ├─────┼─────────┼──────────┼───────────┤
    │ 1   │ 2.87357 │ 0.090045 │ 1         │

    3. Long-format dataframe

    >>> data = Pingouin.read_dataset("rm_anova2")
    >>> Pingouin.homoscedasticity(data, dv="Performance", group="Time")
    1×3 DataFrame
    │ Row │ W       │ pval      │ equal_var │
    │     │ Float64 │ Float64   │ Bool      │
    ├─────┼─────────┼───────────┼───────────┤
    │ 1   │ 3.1922  │ 0.0792169 │ 1         │
"""
function homoscedasticity(data; dv=nothing, group=nothing, method::String="levene", α::Float64=0.05)
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
