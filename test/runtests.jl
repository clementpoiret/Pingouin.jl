using DataFrames
using Pingouin
using Test

@testset "Normality" begin
    x = [2.3, 5.1, 4.3, 2.6, 7.8, 9.2, 1.4]
    res = Pingouin.anderson(x, "norm")
    @test res == (false, 8.571428568870942e-5)

    x = Pingouin.read_dataset("anova")
    res = Pingouin.normality(x[!, "Pain threshold"])
    @test res[1, :W] ≈ 0.9712035839865627
    @test res[1, :pval] ≈ 0.8002574277219071
    @test res[1, :normal] == true
end


@testset "Homoscedasticity" begin
    x = [[4, 8, 9, 20, 14], [5, 8, 15, 45, 12]]
    res = Pingouin.homoscedasticity(x, method = "bartlett", α = 0.05)
    @test res[1, :T] ≈ 2.873568805401419
    @test res[1, :pval] ≈ 0.09004499548422346
    @test res[1, :equal_var] == true
end


@testset "Sphericity" begin
    x = DataFrame(A = [2.2, 3.1, 4.3, 4.1, 7.2],
        B = [1.1, 2.5, 4.1, 5.2, 6.4],
        C = [8.2, 4.5, 3.4, 6.2, 7.2])
    res = Pingouin.sphericity(x)
    @test res[1, :W] ≈ 0.21037236672590112
    @test res[1, :pval] ≈ 0.09649016283209626
end


@testset "Effsize" begin
    x = [1, 2, 3, 4]
    y = [3, 4, 5, 6, 7]
    ef = Pingouin.compute_effsize(x, y, paired = false, eftype = "cohen")
    @test ef ≈ -1.707825127659933
    eta = Pingouin.convert_effsize(ef, "cohen", "eta-square")
    @test eta ≈ 0.42168674698795183
end


@testset "Harrell-Davis" begin
    x = [1.0 2.0 5.0; 2.1 5.0 2.2]

    res = round.(Pingouin.harrelldavis(x, 0.5, 1), digits = 2)
    @test res == [1.55 3.5 3.6]

    x = [1.0 2.0 5.0; 2.1 5.0 2.2]
    res = Pingouin.harrelldavis(x, 0.5, 2)
    @test res[1] ≈ 2.51851852
    @test res[2] ≈ 2.9

    x = [1.0 2.0 5.0; 2.1 5.0 2.2]
    res = Pingouin.harrelldavis(x, [0.25, 0.5, 0.75], 1)
    @test res[1] ≈ [1.16536136 2.45098551 2.62091981]
    @test res[2] ≈ [1.55 3.5 3.6]
    @test res[3] ≈ [1.93463864 4.54901449 4.57908019]
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
        alternative = "two-sided",
        method = "wetzels",
        kappa = 1.0)
    @test bf ≈ 8.221440974059899
end
