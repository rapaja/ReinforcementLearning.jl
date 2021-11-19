@testset "init_Q_from_V" begin
    V = rand(5)
    Q = FMDP.create_Q_from_V(V, 3)
    @test all(size(Q) .== (5, 3))
end