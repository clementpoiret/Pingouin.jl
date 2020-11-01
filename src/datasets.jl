using CSV
using Distributions


"""
    read_dataset(dname)

Read example datasets.

Arguments
----------
- `dname::String`: Name of dataset to read (without extension). Must be a valid dataset present in Pingouin.datasets

Returns
-------
- data : `DataFrame`: Requested dataset.

Examples
--------
Load the `Penguin <https://github.com/allisonhorst/palmerpenguins>`
dataset:

```julia-repl
julia> data = Pingouin.read_dataset("penguins")
344×7 DataFrame
│ Row │ species │ island │ bill_length_mm │ bill_depth_mm │ flipper_length_mm │ body_mass_g │ sex    │
│     │ String  │ String │ String         │ String        │ String            │ String      │ String │
├─────┼─────────┼────────┼────────────────┼───────────────┼───────────────────┼─────────────┼────────┤
│ 1   │ Adelie  │ Biscoe │ 37.8           │ 18.3          │ 174               │ 3400        │ female │
│ 2   │ Adelie  │ Biscoe │ 37.7           │ 18.7          │ 180               │ 3600        │ male   │
│ 3   │ Adelie  │ Biscoe │ 35.9           │ 19.2          │ 189               │ 3800        │ female │
⋮
│ 341 │ Gentoo  │ Biscoe │ 46.8           │ 14.3          │ 215               │ 4850        │ female │
│ 342 │ Gentoo  │ Biscoe │ 50.4           │ 15.7          │ 222               │ 5750        │ male   │
│ 343 │ Gentoo  │ Biscoe │ 45.2           │ 14.8          │ 212               │ 5200        │ female │
│ 344 │ Gentoo  │ Biscoe │ 49.9           │ 16.1          │ 213               │ 5400        │ male   │
```
"""
function read_dataset(dname::String)::DataFrame
    path = joinpath(dirname(@__FILE__), "..", "datasets", "datasets.csv")
    dts = DataFrame(CSV.File(path))
    
    @assert dname in dts[!, :dataset]

    path = joinpath(dirname(@__FILE__), "..", "datasets", "$dname.csv")
    return DataFrame(CSV.File(path))
end


"""
    list_dataset()

List available example datasets.

Returns
-------
- datasets : `DataFrame`: A dataframe with the name, description and reference of all the datasets included in Pingouin.

Examples
--------
```julia-repl
julia> all_datasets = Pingouin.list_dataset()
28×4 DataFrame. Omitted printing of 1 columns
│ Row │ dataset    │ description                                                                                                        │ useful                 │
│     │ String     │ String                                                                                                             │ String                 │
├─────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────────────────┤
│ 1   │ ancova     │ Teaching method with family income as covariate                                                                    │ ANCOVA                 │
│ 2   │ anova      │ Pain threshold per hair color                                                                                      │ anova - pairwise_tukey │
│ 3   │ anova2     │ Fertilizer impact on the yield of crops                                                                            │ anova                  │
⋮
│ 25  │ rm_anova2  │ Performance of employees at two time points and three areas                                                        │ rm_anova2              │
│ 26  │ rm_corr    │ Repeated measurements of pH and PaCO2                                                                              │ rm_corr                │
│ 27  │ rm_missing │ Missing values in long-format repeated measures dataframe                                                          │ rm_anova - rm_anova2   │
│ 28  │ tips       │ One waiter recorded information about each tip he received over a period of a few months working in one restaurant │ regression             │
``` 
"""
function list_dataset()::DataFrame
    path = joinpath(dirname(@__FILE__), "..", "datasets", "datasets.csv")
    return DataFrame(CSV.File(path))
end
