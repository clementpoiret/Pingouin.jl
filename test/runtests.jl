using DataFrames
using Pingouin
using Test

@testset "Normality" begin
    x = [2.3, 5.1, 4.3, 2.6, 7.8, 9.2, 1.4]
    res = Pingouin.anderson(x, dist="norm")
    @test res == (false, 8.571428568870942e-5)

    x = Pingouin.read_dataset("anova")
    res = Pingouin.normality(x[!, "Pain threshold"])
    @test res[1, :W] ≈ -0.8425410994541409
    @test res[1, :pval] ≈ 0.8002574277219333
    @test res[1, :normal] == true
end


@testset "Homoscedasticity" begin
    x = [[4, 8, 9, 20, 14], [5, 8, 15, 45, 12]]
    res = Pingouin.homoscedasticity(x, method="bartlett", α=.05)
    @test res[1, :T] ≈ 2.873568805401419
    @test res[1, :pval] ≈ 0.09004499548422346
    @test res[1, :equal_var] == true
end


@testset "Sphericity" begin
    x = DataFrame(A=[2.2, 3.1, 4.3, 4.1, 7.2],
                  B=[1.1, 2.5, 4.1, 5.2, 6.4],
                  C=[8.2, 4.5, 3.4, 6.2, 7.2])
    res = Pingouin.sphericity(x)
    @test res[1, :W] ≈ 0.21037236672590112
    @test res[1, :pval] ≈ 0.09649016283209626
end


@testset "Effsize" begin   
    x = [1, 2, 3, 4]
    y = [3, 4, 5, 6, 7]
    ef = Pingouin.compute_effsize(x, y, paired=false, eftype="cohen")
    @test ef ≈ -1.707825127659933
    eta = Pingouin.convert_effsize(ef, "cohen", "eta-square")
    @test eta ≈ 0.42168674698795183

end

@testset "Bayes_Factor_Binomial_Test" begin
    bf = Pingouin.bayesfactor_binom(115, 200, 0.5)
    @test bf ≈ 0.8353281455069181
end


@testset "Bayes_Factor_Pearson_Correlation" begin
    r, n = 0.6, 20
    bf = Pingouin.bayesfactor_pearson(r, n)
    @test bf ≈ 10.633616334136537

    bf = Pingouin.bayesfactor_pearson(r, n,
                                      tail="two-sided",
                                      method="wetzels",
                                      kappa=1.)
    @test bf ≈ 8.221440974059899
end
