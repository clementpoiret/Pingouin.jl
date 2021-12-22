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
    estimate::Number;
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

"""
    remove_na(x[, y, paired, axis])

Remove missing values along a given axis in one or more (paired) arrays.

Arguments
---------
- `x::Array{<:Number}`: 1D or 2D array.
- `y::Array{<:Number}`: 1D or 2D array. Must have the same shape as `x`.
- `paired::Bool`: Indicates if the measurements are paired.
- `axis::String`: `rows` or `cols` to filter when `x` is a 2D array.

Returns
-------
- `x::Array{<:Number}`: Filtered array.
- `y::Array{<:Number}`: Filtered array.

Examples
--------
1. Single 1D array

```julia-repl
julia> using Pingouin
julia> x = [6.4, 3.2, 4.5, NaN];
julia> Pingouin.remove_na(x)
3-element Vector{Float64}:
 6.4
 3.2
 4.5
```

2. Single 2D array

```julia-repl
julia> x = [6.4 3.2 4.5 NaN; 5.4 4.0 0.2 6.3; NaN 4.0 0.8 9.1];
julia> Pingouin.remove_na(x, axis="rows")
1×4 Matrix{Float64}:
 5.4  4.0  0.2  6.3
julia> Pingouin.remove_na(x, axis="cols")
3×2 Matrix{Float64}:
 3.2  4.5
 4.0  0.2
 4.0  0.8
```

3. With two paired 1D arrays

```julia-repl
julia> x = [6.4, 3.2, 4.5, NaN];
julia> y = [2.3, NaN, 5.2, 4.6];
julia> Pingouin.remove_na(x, y, paired=true)
([6.4, 4.5], [2.3, 5.2])
```

4. With two independent 2D arrays

```julia-repl
julia> x = [4 2; 4 NaN; 7 6];
julia> y = [6 NaN; 3 2; 2 2];
julia> x_no_nan, y_no_nan = Pingouin.remove_na(x, y, paired=false);
julia> x_no_nan
2×2 Matrix{Float64}:
 4.0  2.0
 7.0  6.0
julia> y_no_nan
2×2 Matrix{Float64}:
 3.0  2.0
 2.0  2.0
```

5. With two paired 2D arrays

```julia-repl
julia> x_no_nan, y_no_nan = Pingouin.remove_na(x, y, paired=true);
julia> x_no_nan
1×2 Matrix{Float64}:
 7.0  6.0
julia> y_no_nan
1×2 Matrix{Float64}:
 2.0  2.0
julia> x_no_nan, y_no_nan = Pingouin.remove_na(x, y, paired=true, axis="cols");
julia> x_no_nan
3×1 Matrix{Float64}:
 4.0
 4.0
 7.0
julia> y_no_nan
3×1 Matrix{Float64}:
 6.0
 3.0
 2.0
```
"""
# 1D
function remove_na(x::Vector{<:Number})::Vector{<:Number}
    @assert length(x) > 0 "Input array must have at least one element."

    return _remove_na_single(x)
end
function remove_na(x::Vector{<:Number},
    y::Union{<:Number,AbstractString})::Tuple{Vector{<:Number},Union{<:Number,AbstractString}}

    @assert length(x) > 0 "Input array must have at least one element."

    return _remove_na_single(x), y
end
function remove_na(x::Vector{<:Number},
    y::Vector{<:Number};
    paired::Bool)::Tuple{Vector{<:Number},Vector{<:Number}}

    @assert length(x) > 0 "Input array must have at least one element."

    if length(y) == 1
        return _remove_na_single(x), y
    elseif (size(x) != size(y)) | !paired
        return _remove_na_single(x), _remove_na_single(y)
    end

    # Let's assume it's paired with equal sizes
    @assert size(x) == size(y) "x and y arrays must have the same shape."

    x_mask = @. !isnan(x)
    y_mask = @. !isnan(y)
    both = x_mask .& y_mask

    return x[vec(both)], y[vec(both)]
end
# 2D
function remove_na(x::Array{<:Number,2}; axis::String = "rows")::Array{<:Number,2}
    @assert length(x) > 0 "Input array must have at least one element."

    return _remove_na_single(x, axis = axis)
end
function remove_na(x::Array{<:Number,2},
    y::Array{<:Number,2};
    paired::Bool,
    axis::String = "rows")::Tuple{Array{<:Number,2},Array{<:Number,2}}

    @assert length(x) > 0 "Input array must have at least one element."
    @assert axis in ["rows", "cols"] "Axis must be \"rows\" or \"cols\"."

    if length(y) == 1
        return _remove_na_single(x, axis = axis), y
    elseif (size(x) != size(y)) | !paired
        return _remove_na_single(x, axis = axis), _remove_na_single(y, axis = axis)
    end

    # Let's assume it's paired with equal sizes
    @assert size(x) == size(y) "x and y arrays must have the same shape."

    dim = axis == "rows" ? 2 : 1
    x_mask = any(isnan.(x), dims = dim)
    y_mask = any(isnan.(y), dims = dim)
    both = @. !x_mask & !y_mask

    if axis == "rows"
        return x[vec(both), :], y[vec(both), :]
    else
        return x[:, vec(both)], y[:, vec(both)]
    end
end
