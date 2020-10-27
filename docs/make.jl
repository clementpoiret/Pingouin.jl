push!(LOAD_PATH, "../src/")

using Documenter
using Pingouin

makedocs(
    sitename="Pingouin",
    format=Documenter.HTML(),
    pages=[
        "index.md",
        "Datasets" => "datasets.md",
        "Distributions" => "distributions.md",
        "Effect Sizes" => "effsize.md",
        "Bayesian" => "bayesian.md",
        "Non-Parametric Tests" => "nonparametric.md"
    ],
    modules=[Pingouin]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#= deploydocs(
    repo = "<repository url>"
) =#
