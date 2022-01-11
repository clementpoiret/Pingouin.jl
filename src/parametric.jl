include("bayesian.jl")
include("effsize.jl")
include("utils.jl")

using DataFrames
using Distributions
using HypothesisTests
using Statistics

"""
    ttest(x, y[, paired, tail, correction, r, confidence])

T-test.

Arguments
---------
- `x::Vector{<:Number}`: First set of observations.
- `y::Union{Float64,Vector{<:Number}}`: Second set of observations. If `y` is a single value, a one-sample T-test is computed against that value (= "mu" in the t.test R function).
- `paired::Bool`: Specify whether the two observations are related (i.e. repeated measures) or independent.
- `tail::Symbol`:  Defines the alternative hypothesis, or tail of the test. Must be one of :both (default), :left or :right. Both :left and :right return one-sided p-values. :right tests against the alternative hypothesis that the mean of `x` is greater than the mean of `y`.
- `correction::Union{String,Bool}`: For unpaired two sample T-tests, specify whether or not to correct for unequal variances using Welch separate variances T-test. If 'auto', it will automatically uses Welch T-test when the sample sizes are unequal, as recommended by Zimmerman 2004.
- `r::Float64`: Cauchy scale factor for computing the Bayes Factor. Smaller values of r (e.g. 0.5), may be appropriate when small effect sizes are expected a priori; larger values of r are appropriate when large effect sizes are expected (Rouder et al 2009). The default is 0.707 (= :math:`\\sqrt{2} / 2`).
- `confidence::Float64`: Confidence level for the confidence intervals (0.95 = 95%)


Returns
-------
stats : :py:class:`pandas.DataFrame`

    * ``'T'``: T-value
    * ``'dof'``: degrees of freedom
    * ``'tail'``: alternative of the test
    * ``'p-val'``: p-value
    * ``'CI95%'``: confidence intervals of the difference in means
    * ``'cohen-d'``: Cohen d effect size
    * ``'BF10'``: Bayes Factor of the alternative hypothesis
    * ``'power'``: achieved power of the test ( = 1 - type II error)

See also
--------
mwu, wilcoxon, anova, rm_anova, pairwise_ttests, compute_effsize

Notes
-----
Missing values are automatically removed from the data. If ``x`` and
``y`` are paired, the entire row is removed (= listwise deletion).

The **T-value for unpaired samples** is defined as:

.. math::

    t = \\frac{\\overline{x} - \\overline{y}}
    {\\sqrt{\\frac{s^{2}_{x}}{n_{x}} + \\frac{s^{2}_{y}}{n_{y}}}}

where :math:`\\overline{x}` and :math:`\\overline{y}` are the sample means,
:math:`n_{x}` and :math:`n_{y}` are the sample sizes, and
:math:`s^{2}_{x}` and :math:`s^{2}_{y}` are the sample variances.
The degrees of freedom :math:`v` are :math:`n_x + n_y - 2` when the sample
sizes are equal. When the sample sizes are unequal or when
:code:`correction=True`, the Welch–Satterthwaite equation is used to
approximate the adjusted degrees of freedom:

.. math::

    v = \\frac{(\\frac{s^{2}_{x}}{n_{x}} + \\frac{s^{2}_{y}}{n_{y}})^{2}}
    {\\frac{(\\frac{s^{2}_{x}}{n_{x}})^{2}}{(n_{x}-1)} +
    \\frac{(\\frac{s^{2}_{y}}{n_{y}})^{2}}{(n_{y}-1)}}

The p-value is then calculated using a T distribution with :math:`v`
degrees of freedom.

The T-value for **paired samples** is defined by:

.. math:: t = \\frac{\\overline{x}_d}{s_{\\overline{x}}}

where

.. math:: s_{\\overline{x}} = \\frac{s_d}{\\sqrt n}

where :math:`\\overline{x}_d` is the sample mean of the differences
between the two paired samples, :math:`n` is the number of observations
(sample size), :math:`s_d` is the sample standard deviation of the
differences and :math:`s_{\\overline{x}}` is the estimated standard error
of the mean of the differences. The p-value is then calculated using a
T-distribution with :math:`n-1` degrees of freedom.

The scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor is approximated
using the :py:func:`pingouin.bayesfactor_ttest` function.

Results have been tested against JASP and the `t.test` R function.

References
----------
* https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm

* Delacre, M., Lakens, D., & Leys, C. (2017). Why psychologists should
    by default use Welch’s t-test instead of Student’s t-test.
    International Review of Social Psychology, 30(1).

* Zimmerman, D. W. (2004). A note on preliminary tests of equality of
    variances. British Journal of Mathematical and Statistical
    Psychology, 57(1), 173-181.

* Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
    2009. Bayesian t tests for accepting and rejecting the null
    hypothesis. Psychon. Bull. Rev. 16, 225–237.
    https://doi.org/10.3758/PBR.16.2.225

Examples
--------
1. One-sample T-test.

>>> from pingouin import ttest
>>> x = [5.5, 2.4, 6.8, 9.6, 4.2]
>>> ttest(x, 4).round(2)
            T  dof alternative  p-val         CI95%  cohen-d   BF10  power
T-test  1.4    4   two-sided   0.23  [2.32, 9.08]     0.62  0.766   0.19

2. One sided paired T-test.

>>> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
>>> post = [6.4, 3.4, 6.4, 11., 4.8]
>>> ttest(pre, post, paired=True, alternative='less').round(2)
            T  dof alternative  p-val          CI95%  cohen-d   BF10  power
T-test -2.31    4        less   0.04  [-inf, -0.05]     0.25  3.122   0.12

Now testing the opposite alternative hypothesis

>>> ttest(pre, post, paired=True, alternative='greater').round(2)
            T  dof alternative  p-val         CI95%  cohen-d  BF10  power
T-test -2.31    4     greater   0.96  [-1.35, inf]     0.25  0.32   0.02

3. Paired T-test with missing values.

>>> import numpy as np
>>> pre = [5.5, 2.4, np.nan, 9.6, 4.2]
>>> post = [6.4, 3.4, 6.4, 11., 4.8]
>>> ttest(pre, post, paired=True).round(3)
            T  dof alternative  p-val          CI95%  cohen-d   BF10  power
T-test -5.902    3   two-sided   0.01  [-1.5, -0.45]    0.306  7.169  0.073

Compare with SciPy

>>> from scipy.stats import ttest_rel
>>> np.round(ttest_rel(pre, post, nan_policy="omit"), 3)
array([-5.902,  0.01 ])

4. Independent two-sample T-test with equal sample size.

>>> np.random.seed(123)
>>> x = np.random.normal(loc=7, size=20)
>>> y = np.random.normal(loc=4, size=20)
>>> ttest(x, y)
                T  dof alternative         p-val         CI95%   cohen-d       BF10  power
T-test  9.106452   38   two-sided  4.306971e-11  [2.64, 4.15]  2.879713  1.366e+08    1.0

5. Independent two-sample T-test with unequal sample size. A Welch's T-test is used.

>>> np.random.seed(123)
>>> y = np.random.normal(loc=6.5, size=15)
>>> ttest(x, y)
                T        dof alternative     p-val          CI95%   cohen-d   BF10     power
T-test  1.996537  31.567592   two-sided  0.054561  [-0.02, 1.65]  0.673518  1.469  0.481867

6. However, the Welch's correction can be disabled:

>>> ttest(x, y, correction=False)
                T  dof alternative     p-val          CI95%   cohen-d   BF10     power
T-test  1.971859   33   two-sided  0.057056  [-0.03, 1.66]  0.673518  1.418  0.481867

Compare with SciPy

>>> from scipy.stats import ttest_ind
>>> np.round(ttest_ind(x, y, equal_var=True), 6)  # T value and p-value
array([1.971859, 0.057056])
"""
function ttest(x::Vector{<:Number},
    y::Float64;
    paired::Bool = false,
    tail::String = :both,
    correction::String = "auto",
    r::Float64 = 0.707,
    confidence::Float64 = 0.95)::DataFrame

    ttest(x, [y],
        paired = paired,
        tail = tail,
        correction = correction,
        r = r,
        confidence = confidence)
end
function ttest(x::Vector{<:Number},
    y::Vector{<:Number};
    paired::Bool = false,
    tail::String = :both,
    correction::Union{Bool,String} = "auto",
    r::Float64 = 0.707,
    confidence::Float64 = 0.95)::DataFrame

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."


    if (size(x) != size(y)) & paired
        throw(DomainError("Paired t-test requires equal sample sizes"))
    end

    # Remove rows with missing values
    x, y = remove_na(x, y, paired = paired)
    nx, ny = length(x), length(y)

    if ny == 1
        # Case one sample T-test
        test = OneSampleTTest(x, y)
        tval = test.t
        pval = pvalue(test, tail = tail)
        ddof = test.df
        se = test.stderr
    end

    if ny > 1 & paired
        # Case paired two samples T-test
        # Do not compute if two arrays are identical
        if x == y
            @warn "x and y are equals. Cannot compute T or p-value."
            tval = pval = se = bf = NaN
            ddof = nx - 1
        else
            test = OneSampleTTest(x, y)
            tval = test.t
            pval = pvalue(test, tail = tail)
            ddof = test.df
            se = test.stderr
        end
    elseif ny > 1 & !paired
        # Case unpaired two samples T-test
        if ((correction == true) | (correction == "auto")) & (nx != ny)
            # Use the Welch separate variance T-test
            test = UnequalVarianceTTest(x, y)
            # ddof are approximated using Welch–Satterthwaite equation            
        else
            test = EqualVarianceTTest(x, y)
        end
        tval = test.t
        pval = pvalue(test, tail = tail)
        ddof = test.df
        se = test.stderr
    end

    # Effect size
    d = compute_effsize(x, y, paired = paired, eftype = "cohen")

    # Confidence interval for the (difference in) means
    # Compare to the t.test r function
    if tail == :both
        α = 1 - confidence
        conf = 1 - α / 2
    else
        conf = confidence
    end
    tcrit = quantile(TDist(ddof), conf)
    ci = [tval - tcrit, tval + tcrit] .* se
    if ny == 1
        ci .+= y
    end

    if tail == :right
        ci[2] = Inf
    elseif tail == :left
        ci[1] = -Inf
    end

    ci_name = "CI$(Int(confidence*100))%"

    # todo: Achieved power
    power = NaN

    # Bayes factor
    bf = bayesfactor_ttest(tval, nx, ny,
        paired = paired,
        tail = tail,
        r = r)

    return DataFrame("dof" => ddof,
        "T" => tval,
        "p-val" => pval,
        "tail" => tail,
        "cohen's d" => abs(d),
        ci_name => [ci],
        "power" => power,
        "BF10" => bf)
end
