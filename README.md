# Pingouin.jl

*A reimplementation of Raphaelvallat's Pingouin in Julia, from scratch.
Currently at a really early stage, not usable at all.*

I'm a PhD student who has to do statistical analysis. I'm also interested in
Julia. To learn Julia, I decided to reimplement my favorite stats lib I used in
Python. I'm starting with the functions I use the most, and simple statistical
tests. I'm open to every suggestions/contributions :)

I'm just starting in Julia, so if you find my code ugly, of if you want to suggest
some good practices, feel free to open an issue <3

Pingouin.jl is an open-source statistical package written in pure Julia,
and based mostly on DataFrames.jl, and HypothesisTests.jl. Some of its main
features are listed below. For a full list of future functions, please refer
to the original Python API documentation.

Pingouin is designed for users who want simple yet exhaustive stats functions:
![sample code](code.png)

## Current progress

- [x] Distribution,
- [x] Effect sizes,
- [ ] **Non-parametric [WIP]**,
- [ ] ANOVA and T-test,
- [ ] Correlation and regression,
- [ ] Multiple comparisons and post-hoc tests,
- [ ] Bayesian,
- [ ] Circular,
- [ ] Contingency,
- [ ] Multivariate tests,
- [ ] Plotting,
- [ ] Power analysis,
- [ ] Reliability and consistency,
- [ ] Others.
