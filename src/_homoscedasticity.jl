using Distributions
using Statistics

function levene(samples; center::String="median", α::Float64=0.05)::Tuple{Bool,Float64,Float64}
    k = length(samples)
    if k < 2
        throw(DomainError(samples, "Must enter at least two input sample arrays."))
    end
    Ni = zeros(k)
    Yci = zeros(k)

    if !(center in ["mean","median","trimmed"])
        throw(DomainError(samples, "Invalid value for argument <center>."))
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

function bartlett(samples; α::Float64=0.05)::Tuple{Bool,Float64,Float64}
    k = length(samples)
    if k < 2
        throw(DomainError(samples, "Must enter at least two input sample arrays."))
    end

    for sample in samples
        if length(sample) == 0
            throw(DomainError(samples, "Samples can't be empty."))
        end
    end

    Ni = zeros(k)
    ssq = zeros(k)
    for j in 1:k
        Ni[j] = length(samples[j])
        ssq[j] = var(samples[j])
    end
    Ntot = sum(Ni)
    spsq = sum((Ni .- 1) .* ssq) ./ (Ntot .- k)
    numer = (Ntot .- k) .* log(spsq) .- sum((Ni .- 1.0) .* (@. log(ssq)))
    denom = 1.0 + 1.0 / (3 * (k - 1)) * ((sum(1.0 ./ (Ni .- 1.0))) - 1.0 / (Ntot - k))

    T = numer / denom
    d = Chisq(k - 1)
    P = 1 - cdf(d, T)
    H = (α >= P)

    return H, T, P
end