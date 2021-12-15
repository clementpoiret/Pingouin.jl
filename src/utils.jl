using DataFrame

"""
    _perm_pval(bootstat, estimate[, alternative])

Compute p-values from a permutation test.

Arguments
---------
- `bootstat::Array{<:Number}`: Permation distribution.
- `estimate::Number`: Point estimate.
- `alternative::String`: Tail for p-value. One of "two-sided" (default), "less", "greater".

Returns
-------
- `pval::Float64`: P-value.
"""
function _perm_pval(bootstat::Array{<:Number},
    estimate::Number,
    alternative::String = "two-sided")::Float64

    @assert alternative in ["two-sided", "less", "greater"] "Alternative must be \"two-sided\", \"less\" or \"greater\"."

    n_boot = length(bootstat)
    @assert n_boot > 0 "Bootstrap distribution must have at least one element."

    if (alternative == "greater")
        pval = sum(bootstat .>= estimate) / n_boot
    elseif (alternative == "less")
        pval = sum(bootstat .<= estimate) / n_boot
    else
        pval = sum(abs.(bootstat) .>= abs(estimate)) / n_boot
    end

    return pval
end
