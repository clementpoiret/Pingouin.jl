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

```julia-repl
julia> import Pingouin.ttest
julia> x = [5.5, 2.4, 6.8, 9.6, 4.2]
julia> ttest(x, 4)
1×8 DataFrame
 Row │ dof    T        p-val     tail    cohen's d  CI95%               power     BF10     
     │ Int64  Float64  Float64   Symbol  Float64    Array…              Float64   Float64  
─────┼─────────────────────────────────────────────────────────────────────────────────────
   1 │     4  1.39739  0.234824  both     0.624932  [2.32231, 9.07769]  0.191796  0.766047
```

2. One sided paired T-test.

```julia-repl
julia> pre = [5.5, 2.4, 6.8, 9.6, 4.2]
julia> post = [6.4, 3.4, 6.4, 11., 4.8]
julia> ttest(pre, post, paired=true, tail=:left)
1×8 DataFrame
 Row │ dof    T         p-val     tail    cohen's d  CI95%               power    BF10    
     │ Int64  Float64   Float64   Symbol  Float64    Array…              Float64  Float64 
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │     4  -2.30783  0.041114  left     0.250801  [-Inf, -0.0533789]  0.12048  3.12204


# Now testing the opposite alternative hypothesis
julia> ttest(pre, post, paired=true, tail=:right)
1×8 DataFrame
 Row │ dof    T         p-val     tail    cohen's d  CI95%            power      BF10     
     │ Int64  Float64   Float64   Symbol  Float64    Array…           Float64    Float64  
─────┼────────────────────────────────────────────────────────────────────────────────────
   1 │     4  -2.30783  0.958886  right    0.250801  [-1.34662, Inf]  0.0168646  0.320303
```

3. Paired T-test with missing values.

```julia-repl
julia> pre = [5.5, 2.4, NaN, 9.6, 4.2]
julia> post = [6.4, 3.4, 6.4, 11., 4.8]
julia> ttest(pre, post, paired=true)
1×8 DataFrame
 Row │ dof    T         p-val       tail    cohen's d  CI95%                  power      BF10    
     │ Int64  Float64   Float64     Symbol  Float64    Array…                 Float64    Float64 
─────┼───────────────────────────────────────────────────────────────────────────────────────────
   1 │     3  -5.90187  0.00971277  both     0.306268  [-1.50075, -0.449254]  0.0729667  7.16912
```

4. Independent two-sample T-test with equal sample size.

```julia-repl
julia> using Random
julia> x = rand(Float64, 20) .+ 5
julia> y = rand(Float64, 20) .+ 4
julia> ttest(x, y)
1×8 DataFrame
 Row │ dof    T        p-val        tail    cohen's d  CI95%                power    BF10       
     │ Int64  Float64  Float64      Symbol  Float64    Array…               Float64  Float64    
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │    38  11.3107  9.99425e-14  both      3.57675  [0.856537, 1.22998]      1.0  4.27574e10
```

5. Independent two-sample T-test with unequal sample size. A Welch's T-test is used.

```julia-repl
julia> y = rand(Float64, 15) .+ 4
julia> ttest(x, y)
1×8 DataFrame
 Row │ dof      T        p-val        tail    cohen's d  CI95%              power    BF10      
     │ Float64  Float64  Float64      Symbol  Float64    Array…             Float64  Float64   
─────┼─────────────────────────────────────────────────────────────────────────────────────────
   1 │ 29.7616  11.0515  4.69414e-12  both       3.7929  [0.9018, 1.31082]      1.0  3.68389e9
```

6. However, the Welch's correction can be disabled:

```julia-repl
julia> ttest(x, y, correction=false)
1×8 DataFrame
 Row │ dof    T        p-val        tail    cohen's d  CI95%              power    BF10      
     │ Int64  Float64  Float64      Symbol  Float64    Array…             Float64  Float64   
─────┼───────────────────────────────────────────────────────────────────────────────────────
   1 │    33  11.1045  1.10246e-12  both       3.7929  [0.903617, 1.309]      1.0  4.14836e9
```
"""
function ttest(x::Vector{<:Number},
    y::Real;
    paired::Bool = false,
    tail::Symbol = :both,
    r::Float64 = 0.707,
    confidence::Float64 = 0.95)::DataFrame

    @assert tail in [:both, :left, :right] "Tail must be one of :both (default), :left or :right."

    x = remove_na(x)
    nx = length(x)

    # Case one sample T-test
    test = OneSampleTTest(x, y)
    tval = test.t
    pval = pvalue(test, tail = tail)
    ddof = test.df
    se = test.stderr

    # Effect size
    d = compute_effsize(x, [y], paired = paired, eftype = "cohen")

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
    ci .+= y

    if tail == :right
        ci[2] = Inf
    elseif tail == :left
        ci[1] = -Inf
    end

    ci_name = "CI$(Int(confidence*100))%"

    # One-sample
    power = power_ttest(d, nx, nothing, 0.05, contrast = "one-sample", tail = tail)

    # Bayes factor
    bf = bayesfactor_ttest(tval,
        nx,
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
function ttest(x::Vector{<:Number},
    y::Vector{<:Number};
    paired::Bool = false,
    tail::Symbol = :both,
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
        ttest(x, y[1], paired = paired, tail = tail, r = r, confidence = confidence)
    end

    if ny > 1 && paired
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
        if (correction == true) || (correction == "auto" && nx != ny)
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

    if tail == :right
        ci[2] = Inf
    elseif tail == :left
        ci[1] = -Inf
    end

    ci_name = "CI$(Int(confidence*100))%"

    # Achieved power
    if ny > 1 && paired
        # Paired two-samples
        power = power_ttest(d, nx, nothing, 0.05, contrast = "paired", tail = tail)
    elseif ny > 1 && !paired
        # Independant two-samples
        if nx == ny
            # Equal sample sizes
            power = power_ttest(d, nx, nothing, 0.05, tail = tail)
        else
            # Unequal sample sizes
            power = power_ttest2n(nx, ny, d, nothing, 0.05, tail = tail)
        end
    end

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
