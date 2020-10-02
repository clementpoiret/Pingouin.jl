using Distributions
using Statistics

function levene(samples; center::String="median", α::Float64=0.05) 
    k = length(samples)
    if k < 2
        throw(DomainError(x, "Must enter at least two input sample arrays."))
    end
    Ni = zeros(k)
    Yci = zeros(k)

    if !(center in ["mean","median","trimmed"])
        throw(DomainError(x, "Invalid value for argument <center>."))
    end

    if center == "median"
        func = median
    elseif center == "mean"
        func = mean
    end

    for j in 1:k
        Ni[j] = length(samples[j])
        Yci[j] = func(samples[j])
    end
    Ntot = sum(Ni)

    # compute Zij's
    Zij = [[] for i in 1:k]
    for i in 1:k
        Zij[i] = @. abs(samples[i] .- Yci[i])
    end

    # compute Zbari
    Zbari = zeros(k)
    Zbar = 0.0
    for i in 1:k
        Zbari[i] = mean(Zij[i])
        Zbar += Zbari[i] * Ni[i]
    end
    Zbar /= Ntot

    numer = (Ntot - k) * sum(Ni .* (Zbari .- Zbar).^2)

    # compute denom_variance
    dvar = 0.0
    for i in 1:k
        dvar += sum((Zij[i] .- Zbari[i]).^2)
    end

    denom = (k - 1.0) * dvar

    W = numer / denom
    d = FDist(k - 1, Ntot - k)
    P = 1 - cdf(d, W)

    # Test the null hypothesis
    H = (α >= P)
    
    return H, W, P
end
