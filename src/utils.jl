using DataFrame

"""
    _perm_pval(bootstat, estimate[, alternative])

Compute p-values from a permutation test.

Arguments
---------
- `bootstat::Vector{<:Number}`: Permation distribution.
- `estimate::Number`: Point estimate.
- `alternative::String`: Tail for p-value. One of "two-sided" (default), "less", "greater".

Returns
-------
- `pval::Float64`: P-value.
"""
function _perm_pval(bootstat::Vector{<:Number},
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


# Missing values
"""
    _remove_na_single(x[, axis])

Remove NaN in a single array.

Arguments
---------
- `x::Array{<:Number}`. Array to filter.
- `axis::String`. `rows` or `cols` to filter when `x` is a 2D array.

Returns
-------
- `x::Array{<:Number}`. Filtered array.
"""
function _remove_na_single(x::Vector{<:Number})::Vector{<:Number}
    return @. x[!isnan(x)]
end
function _remove_na_single(x::Array{<:Number,2}; axis::String = "rows")::Array{<:Number,2}
    dim = axis == "rows" ? 2 : 1
    nan_idx = any(isnan.(x), dims = dim)

    if axis == "rows"
        return x[vec(@. !nan_idx), :]
    else
        return x[:, vec(@. !nan_idx)]
    end
end
