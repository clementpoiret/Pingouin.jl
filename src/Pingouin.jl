module Pingouin

export normality

# todo: rename alternative to tail for consistency w/ HypothesisTests.jl

include("bayesian.jl")
include("datasets.jl")
include("distributions.jl")
include("effsize.jl")
include("nonparametric.jl")
include("power.jl")
include("utils.jl")

end # module
