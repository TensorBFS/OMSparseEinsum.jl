using Test, SparseTN

@testset "patch" begin
    @test BigInt(LongLongUInt((3,))) == 3
    @test BigInt(LongLongUInt((3, 0))) == BigInt(3) << 64
    @test BigInt(LongLongUInt((0, 3))) == 3
    @test BigInt(LongLongUInt((1,2))) == (BigInt(1) << 64) + BigInt(2)
    @test log2i(LongLongUInt((3,))) == 1
    @test log2i(LongLongUInt((1,2))) == 64
    @test log2i(LongLongUInt((0,2))) == 1
    @test log2i(LongLongUInt((1,0))) == 64
end

