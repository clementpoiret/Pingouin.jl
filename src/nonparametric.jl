using Distributions
using StatsBase

"""Robust outlier detection based on the MAD-median rule.

Parameters
----------
a : array-like
    Input array. Must be one-dimensional.

Returns
-------
outliers: boolean (same shape as a)
    Boolean array indicating whether each sample is an outlier (true) or
    not (false).

See also
--------
mad

Notes
-----
The MAD-median-rule ([1]_, [2]_) will refer to declaring :math:`X_i`
an outlier if

.. math::

    \\frac{\\left | X_i - M \\right |}{\\text{MAD}_{\\text{norm}}} > K,

where :math:`M` is the median of :math:`X`,
:math:`\\text{MAD}_{\\text{norm}}` the normalized median absolute deviation
of :math:`X`, and :math:`K` is the square
root of the .975 quantile of a :math:`X^2` distribution with one degree
of freedom, which is roughly equal to 2.24.

References
----------
.. [1] Hall, P., Welsh, A.H., 1985. Limit theorems for the median
   deviation. Ann. Inst. Stat. Math. 37, 27–36.
   https://doi.org/10.1007/BF02481078

.. [2] Wilcox, R. R. Introduction to Robust Estimation and Hypothesis
   Testing. (Academic Press, 2011).

Examples
--------
>>> a = [-1.09, 1., 0.28, -1.51, -0.58, 6.61, -2.43, -0.43]
>>> Pingouin.madmedianrule(a)
8-element Array{Bool,1}:
 0
 0
 0
 0
 0
 1
 0
 0
"""
function madmedianrule(a::Array{<:Number})::Array{Bool}
    @assert length(size(a)) == 1 "Only 1D array / list are supported for this function."
    k = sqrt(quantile(Chisq(1), 0.975))
    return (abs.(a .- median(a)) ./ mad(a)) .> k
end
