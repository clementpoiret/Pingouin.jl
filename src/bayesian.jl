using SciPy
using SpecialFunctions
using HypergeometricFunctions

"""
Bayes factor of a binomial test with :math:`k` successes,
:math:`n` trials and base probability :math:`p`.

Parameters
----------
k : int
    Number of successes.
n : int
    Number of trials.
p : float
    Base probability of success (range from 0 to 1).

Returns
-------
binom_bf : float
    The Bayes Factor quantifies the evidence in favour of the
    alternative hypothesis, where the null hypothesis is that
    the random variable is binomially distributed with base probability
    :math:`p`.

See also
--------
bayesfactor_pearson : Bayes Factor of a correlation
bayesfactor_ttest : Bayes Factor of a T-test

Notes
-----
Adapted from a Matlab code found at
https://github.com/anne-urai/Tools/blob/master/stats/BayesFactors/binombf.m
The Bayes Factor is given by the formula below:
.. math::
    BF_{10} = \\frac{\\int_0^1 \\binom{n}{k}g^k(1-g)^{n-k}}
    {\\binom{n}{k} p^k (1-p)^{n-k}}

References
----------
* http://pcl.missouri.edu/bf-binomial
* https://en.wikipedia.org/wiki/Bayes_factor

Examples
--------
We want to determine if a coin is fair. After tossing the coin 200 times
in a row, we report 115 heads (hereafter referred to as "successes") and 85
tails ("failures"). The Bayes Factor can be easily computed using Pingouin:

> using Pingouin
> bf = Pingouin.bayesfactor_binom(115, 200, 0.5)
> # Note that Pingouin returns the BF-alt by default.
> # BF-null is simply 1 / BF-alt
> print("BF-null: %.3f, BF-alt: %.3f" % (1 / bf, bf))
BF-null: 1.197, BF-alt: 0.835

Since the Bayes Factor of the null hypothesis ("the coin is fair") is
higher than the Bayes Factor of the alternative hypothesis
("the coin is not fair"), we can conclude that there is more evidence to
support the fact that the coin is indeed fair. However, the strength of the
evidence in favor of the null hypothesis (1.197) is "barely worth
mentionning" according to Jeffreys's rule of thumb.

Interestingly, a frequentist alternative to this test would give very
different results. It can be performed using the
`SciPy.stats.binom_test` function:

> using SciPy
> pval = SciPy.stats.binom_test(115, 200, p=0.5)
> round.(pval, digits=5)
0.04004

The binomial test rejects the null hypothesis that the coin is fair at the
5% significance level (p=0.04). Thus, whereas a frequentist hypothesis test
would yield significant results at the 5% significance level, the Bayes
factor does not find any evidence that the coin is unfair.
Last example using a different base probability of successes

> bf = Pingouin.bayesian.bayesfactor_binom(k=100, n=1000, p=0.1)
> print("Bayes Factor: ", round.(bf, digits=3))
Bayes Factor: 0.024
"""

function bayesfactor_binom(k::Int64,
                           n::Int64,
                           p::Float64=.5)::Float64
    @assert(0 < p < 1, "p must be between 0 and 1.")
    @assert(k < n, "k (successes) cannot be higher than n (trials).")

    function f(g, k, n)
        binom_pmf = SciPy.stats.binom.pmf(k, n, g)
        return binom_pmf
    end
    binom_bf = SciPy.integrate.quad(f, 0, 1, args=(k, n))[1] / SciPy.stats.binom.pmf(k, n, p)
    return binom_bf
end


"""
Bayes Factor of a Pearson correlation.

Parameters
----------
r : float
    Pearson correlation coefficient.
n : int
    Sample size.
tail : float
    Tail of the alternative hypothesis. Can be *'two-sided'*,
    *'one-sided'*, *'greater'* or *'less'*. *'greater'* corresponds to a
    positive correlation, *'less'* to a negative correlation.
    If *'one-sided'*, the directionality is inferred based on the ``r``
    value (= *'greater'* if ``r`` > 0, *'less'* if ``r`` < 0).
method : str
    Method to compute the Bayes Factor. Can be *'ly'* (default) or
    *'wetzels'*. The former has an exact analytical solution, while the
    latter requires integral solving (and is therefore slower). *'wetzels'*
    was the default in Pingouin <= 0.2.5. See Notes for details.
kappa : float
    Kappa factor. This is sometimes called the *rscale* parameter, and
    is only used when ``method`` is *'ly'*.

Returns
-------
bf : float
    Bayes Factor (BF10).
    The Bayes Factor quantifies the evidence in favour of the alternative
    hypothesis.

See also
--------
corr : (Robust) correlation between two variables
pairwise_corr : Pairwise correlation between columns of a pandas DataFrame
bayesfactor_ttest : Bayes Factor of a T-test
bayesfactor_binom : Bayes Factor of a binomial test

Notes
-----
To compute the Bayes Factor directly from the raw data, use the
:py:func:`pingouin.corr` function.
The two-sided **Wetzels Bayes Factor** (also called *JZS Bayes Factor*)
is calculated using the equation 13 and associated R code of [1]_:
.. math::
    \\text{BF}_{10}(n, r) = \\frac{\\sqrt{n/2}}{\\gamma(1/2)}*
    \\int_{0}^{\\infty}e((n-2)/2)*
    log(1+g)+(-(n-1)/2)log(1+(1-r^2)*g)+(-3/2)log(g)-n/2g
where :math:`n` is the sample size, :math:`r` is the Pearson correlation
coefficient and :math:`g` is is an auxiliary variable that is integrated
out numerically. Since the Wetzels Bayes Factor requires solving an
integral, it is slower than the analytical solution described below.
The two-sided **Ly Bayes Factor** (also called *Jeffreys
exact Bayes Factor*) is calculated using equation 25 of [2]_:
.. math::
    \\text{BF}_{10;k}(n, r) = \\frac{2^{\\frac{k-2}{k}}\\sqrt{\\pi}}
    {\\beta(\\frac{1}{k}, \\frac{1}{k})} \\cdot
    \\frac{\\Gamma(\\frac{2+k(n-1)}{2k})}{\\Gamma(\\frac{2+nk}{2k})}
    \\cdot 2F_1(\\frac{n-1}{2}, \\frac{n-1}{2}, \\frac{2+nk}{2k}, r^2)
The one-sided version is described in eq. 27 and 28 of Ly et al, 2016.
Please take note that the one-sided test requires the
`mpmath <http://mpmath.org/>`_ package.
Results have been validated against JASP and the BayesFactor R package.

References
----------
.. [1] Ly, A., Verhagen, J. & Wagenmakers, E.-J. Harold Jeffreys’s default
   Bayes factor hypothesis tests: Explanation, extension, and
   application in psychology. J. Math. Psychol. 72, 19–32 (2016).
.. [2] Wetzels, R. & Wagenmakers, E.-J. A default Bayesian hypothesis test
   for correlations and partial correlations. Psychon. Bull. Rev. 19,
   1057–1064 (2012).

Examples
--------
Bayes Factor of a Pearson correlation

> using Pingouin
> r, n = 0.6, 20
> bf = Pingouin.bayesfactor_pearson(r, n)
> print("Bayes Factor: ", round.(bf, digits=3))
Bayes Factor: 10.634

Compare to Wetzels method:

> bf = Pingouin.bayesfactor_pearson(r, n, "two-sided", "wetzels", 1.)
> print("Bayes Factor: ", round.(bf, digits=3))
Bayes Factor: 8.221

One-sided test

> bf10pos = Pingouin.bayesfactor_pearson(r, n, "greater", "ly", 1.0)
> bf10neg = Pingouin.bayesfactor_pearson(r, n, "less", "ly", 1.0)
> print("BF-pos: ", round.(bf10pos, digits=3)," BF-neg: ", round.(bf10neg, digits=3))
BF-pos: 21.185, BF-neg: 0.082

We can also only pass ``tail='one-sided'`` and Pingouin will automatically
infer the directionality of the test based on the ``r`` value.

> print("BF: ", round.(bayesfactor_pearson(r, n, tail='one-sided'), digits=3))
BF: 21.185
"""
function bayesfactor_pearson(r::Float64,
                             n::Int64,
                             tail::String="two-sided",
                             method::String="ly",
                             kappa::Float64=1.)::Float64
    @assert(lowercase(method) in ["ly", "wetzels"], "Method not recognized.")
    @assert(lowercase(tail) in ["two-sided", "one-sided", "greater", "less", "g", "l", "positive", "negative", "pos", "neg"])

    # Wrong input
    if !isfinite(r) || n < 2
        return NaN
    end

    @assert(-1 <= r <= 1, "r must be between -1 and 1.")

    if lowercase(tail) != "two-sided" && lowercase(method) == "wetzels"
        @warn "One-sided Bayes Factor are not supported by the Wetzels's method. Switching to method='ly'."
        method = "ly"
    end

    if lowercase(method) == "wetzels"
        # Wetzels & Wagenmakers, 2012. Integral solving

        function f(g, r, n)
            return exp(((n - 2) / 2) * log(1 + g) + (-(n - 1) / 2)
                       * log(1 + (1 - r^2) * g) + (-3 / 2)
                       * log(g) + - n / (2 * g))
        end

        integr = SciPy.integrate.quad(f, 0, Inf, args=(r, n))[1]
        bf10 = sqrt((n / 2)) / gamma(1 / 2) * integr
    else
        # Ly et al, 2016. Analytical solution.
        k = kappa
        lbeta = logbeta(1 / k, 1 / k)
        log_hyperterm = log(_₂F₁(((n - 1) / 2), ((n - 1) / 2),
                                   ((n + 2 / k) / 2), r^2))
        bf10 = exp((1 - 2 / k) * log(2) + 0.5 * log(pi) - lbeta
                   + loggamma((n + 2 / k - 1) / 2) - loggamma((n + 2 / k) / 2) +
                   log_hyperterm)

        if lowercase(tail) != "two-sided"
            # Directional test.
            # We need mpmath for the generalized hypergeometric function
            hyper_term = float(_₃F₂(1, n / 2, n / 2, 3 / 2,
                                      (2 + k * (n + 1)) / (2 * k),
                                      r^2))
            log_term = 2 * (loggamma(n / 2) - loggamma((n - 1) / 2)) - lbeta
            C = 2^((3 * k - 2) / k) * k * r / (2 + (n - 1) * k) * \
                exp(log_term) * hyper_term

            bf10neg = bf10 - C
            bf10pos = 2 * bf10 - bf10neg
            if lowercase(tail) in ["one-sided"]
                # Automatically find the directionality of the test based on r
                if r>= 0
                    bf10 = bf10pos
                else
                    bf10 = bf10neg
                end
            elseif lowercase(tail) in ["greater", "g", "positive", "pos"]
                # We expect the correlation to be positive
                bf10 = bf10pos
            else
                # We expect the correlation to be negative
                bf10 = bf10neg
            end
        end
    end
    return bf10
end


"""
Bayes Factor of a T-test.

Parameters
----------
t : float
    T-value of the T-test
nx : int
    Sample size of first group
ny : int
    Sample size of second group (only needed in case of an independent
    two-sample T-test)
paired : boolean
    Specify whether the two observations are related (i.e. repeated
    measures) or independent.
tail : string
    Specify whether the test is `'one-sided'` or `'two-sided'`. Can also be
    `'greater'` or `'less'` to specify the direction of the test.
    .. warning:: One-sided Bayes Factor (BF) are simply obtained by
        doubling the two-sided BF, which is not exactly the same behavior
        as R or JASP. Be extra careful when interpretating one-sided BF,
        and if you can, always double-check your results.
r : float
    Cauchy scale factor. Smaller values of ``r`` (e.g. 0.5), may be
    appropriate when small effect sizes are expected a priori; larger
    values of ``r`` are appropriate when large effect sizes are
    expected (Rouder et al 2009). The default is
    :math:`\\sqrt{2} / 2 \\approx 0.707`.

Returns
-------
bf : float
    Scaled Jeffrey-Zellner-Siow (JZS) Bayes Factor (BF10).
    The Bayes Factor quantifies the evidence in favour of the
    alternative hypothesis.

See also
--------
ttest : T-test
pairwise_ttest : Pairwise T-tests
bayesfactor_pearson : Bayes Factor of a correlation
bayesfactor_binom : Bayes Factor of a binomial test

Notes
-----
Adapted from a Matlab code found at
https://github.com/anne-urai/Tools/tree/master/stats/BayesFactors
If you would like to compute the Bayes Factor directly from the raw data
instead of from the T-value, use the :py:func:`pingouin.ttest` function.
The JZS Bayes Factor is approximated using the formula described
in ref [1]_:
.. math::
    \\text{BF}_{10} = \\frac{\\int_{0}^{\\infty}(1 + Ngr^2)^{-1/2}
    (1 + \\frac{t^2}{v(1 + Ngr^2)})^{-(v+1) / 2}(2\\pi)^{-1/2}g^
    {-3/2}e^{-1/2g}}{(1 + \\frac{t^2}{v})^{-(v+1) / 2}}
where :math:`t` is the T-value, :math:`v` the degrees of freedom,
:math:`N` the sample size, :math:`r` the Cauchy scale factor
(= prior on effect size) and :math:`g` is is an auxiliary variable
that is integrated out numerically.
Results have been validated against JASP and the BayesFactor R package.

References
----------
.. [1] Rouder, J.N., Speckman, P.L., Sun, D., Morey, R.D., Iverson, G.,
   2009. Bayesian t tests for accepting and rejecting the null hypothesis.
   Psychon. Bull. Rev. 16, 225–237. https://doi.org/10.3758/PBR.16.2.225

Examples
--------
1. Bayes Factor of an independent two-sample T-test

> from Pingouin
> bf = Pingouin.bayesian.bayesfactor_ttest(3.5, 20, 20)
> print("Bayes Factor: ", round.(bf, digits=3), "(two-sample independent)")
Bayes Factor: 26.743 (two-sample independent)

2. Bayes Factor of a paired two-sample T-test

> bf = Pingouin.bayesian.bayesfactor_ttest(3.5, 20, 20, paired=True)
> print("Bayes Factor: ", round.(bf, digits=3), "(two-sample paired)")
Bayes Factor: 17.185 (two-sample paired)

3. Bayes Factor of an one-sided one-sample T-test

> bf = Pingouin.bayesian.bayesfactor_ttest(3.5, 20, tail='one-sided')
> print("Bayes Factor: ", round.(bf, digits=3), "(one-sample)")
Bayes Factor: 34.369 (one-sample)

4. Now specifying the direction of the test

> tval = -3.5
> bf_greater = Pingouin.bayesian.bayesfactor_ttest(tval, 20, tail='greater')
> bf_less = Pingouin.bayesian.bayesfactor_ttest(tval, 20, tail='less')
> print("BF10-greater: ", round.(bf_greater, digits=3), " | BF10-less: ", round.(bf_less, digits=3))
BF10-greater: 0.029 | BF10-less: 34.369
"""
function bayesfactor_ttest(t::Float64,
                           nx::Int64,
                           ny::Int64=nothing,
                           paired::Bool=false,
                           tail::String="two-sided",
                           r::Float64=.707)::Float64
    # Check tails
    possible_tails = ["two-sided", "one-sided", "greater", "less"]
    @assert(tail in possible_tails, "Invalid tail argument.")
    if ny == nothing || ny == 1
        one_sample = true 
    else
        one_sample = false
    end

    # Check T-value
    @assert(isa(t, Int64) || isa(t, Float64), "The T-value must be a int or a float.")
    if !isfinite(t)
        return NaN
    end

    # Function to be integrated
    function f(g, t, n, r, df)
        return (1 + n * g * r^2)^(-.5) * (1 + t^2 / ((1 + n * g * r^2) * df))^(-(df + 1) / 2) * \
               (2 * π)^(-.5) * g^(-3. / 2) * exp(-1 / (2 * g))
    end

    # Define n and degrees of freedom
    if one_sample || paired
        n = nx
        df = n - 1
    else
        n = nx * ny / (nx + ny)
        df = nx + ny - 2
    end

    # JZS Bayes factor calculation: eq. 1 in Rouder et al. (2009)
    integr = SciPy.integrate.quad(f, 0, Inf, args=(t, n, r, df))[1]
    bf10 = 1 / ((1 + t^2 / df)^(-(df + 1) / 2) / integr)

    # Tail
    if tail == "two-sided"
        tail_binary = "two-sided" 
    else
        tail_binary = "one-sided"
    end

    if tail_binary == "one-sided"
        bf10 = bf10 * (1 / 0.5)
    end
    # Now check the direction of the test
    if ((tail == "greater" && t < 0) || (tail == "less" && t > 0)) && bf10 > 1
        bf10 = 1 / bf10
    end
    return bf10
end
