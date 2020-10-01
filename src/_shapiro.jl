"""
Translated to Julia from:
https://github.com/ExploreASL/ExploreASL/blob/495ecc662cd0fd2c59ebcce3469d615eb6b0a89d/Functions/xASL_stat_ShapiroWilk.m
"""

using Distributions
using SpecialFunctions

"""
ALGORITHM AS R94 APPL. STATIST. (1995) VOL.44, NO.4
ROYSTON, Patrick. Remark AS R94: A remark on algorithm AS 181: The W-test for normality.
Journal of the Royal Statistical Society. Series C (Applied Statistics), 1995, vol. 44, no 4, p. 547-551.

Calculates the shapiro wilk statistic of a given data array.
"""
function shapiro_wilk(x::Array{}, α::Float64=0.05)::Tuple{Bool,Float64,Float64}
    if length(x) < 1 || isempty(x)
        throw(DomainError(x, "'x' is empty."))
    end

    if length(x) < 4 
        throw(DomainError(x, "Data must be at least length 4."))
    end

    @assert 0.0 < α < 1.0

    x = sort(x)
    n = length(x)
    mi = [quantile.(Normal(), (((i) - (3.0 / 8.0)) / (n + .25))) for i in 1:n]
    miSq = sum(mi.^2)

    if kurtosis(x) > 3
        # kurtosis is higher than 3 -> Leptokurtic Distribution
        # Perform Shapiro-Francia test

        weights = mi ./ sqrt(miSq)
        W = (sum(weights .* x)^2) / sum((x .- mean(x)).^2)

        nLog = log(n)
        u1 = log(nLog) - nLog
        u2 = log(nLog) + 2 / nLog
        μ = -1.2725 + 1.0521 * u1
        σ = 1.0308 - 0.26758 * u2
        
        # Normalized Shapiro-Francia statistics 
        SF = (log(1 - W) - μ) / σ
        P = 1 - (0.5 * erfc(-SF ./ sqrt(2)))

        return SF, P
    else
        # kurtosis is lower than 3 -> platykurtic Distribution
        # Perform Shapiro-Wilk
        u = 1 / sqrt(n)
        weights = zeros(n, 1)
        weights[n] = -2.706056 * (u^5) + 4.434685 * (u^4) - 2.071190 * (u^3) - 0.147981 * (u^2) + 0.221157 * u + mi[n] / sqrt(miSq)
        weights[1] = -weights[n]

        if n >= 6
            weights[n - 1] = -3.582633 * (u^5) + 5.682633 * (u^4) - 1.752461 * (u^3) - 0.293762 * (u^2) + 0.042981 * u + mi[n - 1] / sqrt(miSq)
            weights[2]   = -weights[n - 1]
            
            count = 3;
            eps = (miSq - 2 * (mi[n]^2) - 2 * (mi[n - 1]^2)) / (1 - 2 * (weights[n]^2) - 2 * (weights[n - 1]^2))
        else
            count = 2
            eps = (miSq - 2 * (mi[n]^2)) / (1 - 2 * (weights[n]^2))
        end

        weights[count:(n - count + 1)] = mi[count:(n - count + 1)] / sqrt(eps)

        W = (sum(weights .* x)^2) / sum((x .- mean(x)).^2)

        if n <= 11
            μ = -0.0006714 * (n^3) + 0.0250540 * (n^2) - 0.39978 * n + 0.54400
            σ = exp(-0.0020322 * (n^3) + 0.0627670 * (n^2) - 0.77857 * n + 1.38220)
            γ = 0.459 * n - 2.273

            SW = -log(γ - log(1 - W))
        else
            nLog = log(n)
            μ = 0.0038915 * (nLog^3) - 0.083751 * (nLog^2) - 0.31082 * nLog - 1.5861
            σ = exp(0.0030302 * (nLog^2) - 0.082676 * nLog - 0.4803)
        
            SW = log(1 - W)
        end

        # Normalize the Shapiro-Wilk statistics
        SW = (SW - μ) / σ
        # P-value
        P = 1 - (0.5 * erfc(-SW ./ sqrt(2)))
    end

    # Test the null hypothesis
    H = (α >= P)

    return H, SW, P
end