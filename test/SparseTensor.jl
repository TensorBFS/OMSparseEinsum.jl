using BitBasis, SparseArrays, SparseTN
using SparseTN: bpermute, linear2cartesian, cartesian2linear
using Test

@testset "constructor" begin
    sv = SparseVector([1,0,0,1,1,0,0,0])
    @test copy(sv) == sv
    t = SparseTensor(sv, (2, 2, 2))
    @test [t[i] for i in 1:8] == [1,0,0,1,1,0,0,0]
    @test ndims(t) == 3
    @test size(t) == (2,2,2)
    @test nnz(t) == 3

    # linear and cartesian indexing
    @test linear2cartesian(t, 1) == (1,1,1)
    @test linear2cartesian(t, 2) == (2,1,1)
    @test linear2cartesian(t, 3) == (1,2,1)
    @test cartesian2linear(t, (1,1,1)) == 1
    @test cartesian2linear(t, (1,1,2)) == 5
    @test cartesian2linear(t, (1,2,1)) == 3

    # errors
    sv = SparseVector([1,0,0,1,1,0,0,0,1])
    t = SparseTensor(sv, (3,3))
    @test_throws AssertionError SparseTensor(sv, (3, 4, 2))

    # printing
    println(t)

    # zeros
    @test zero(t) == zeros(size(t))
    t = stzeros(Float64, (2,2,2,2,2))
    @test size(t) == (2,2,2,2,2)
    @test eltype(t) == Float64
end

@testset "indexing" begin
    sv = SparseVector([1,0,0,1,1,0,0,0])
    t = SparseTensor(sv, (2, 2, 2))
    t2 = SparseTensor(Array(t))
    @show t2
    @test t2 == t
    @test nnz(t2) == 3
    @test_throws BoundsError size(t,-1)
    @test size(t,3) == 2
    @test size(t,4) == 1
    @test size(t) == (2,2,2)
    cis = CartesianIndices((2,2,2))
    @test t[1] == 1
    @test t[2] == t[2,1,1] == 0 == t[cis[2,1,1]]
    @test t[5] == t[1,1,2] == 1 == t[cis[1,1,2]]

    # setindex!
    m = zeros(Int, 2,2,2); m[[1,4,5]] .= 1
    @test Array(t) == m
    @test vec(t) == vec(m)
    t[2] = 8
    @test t[2] == 8
    t[2] = 7
    @test t[2] == 7
    @test zero(t) isa SparseTensor && nnz(zero(t)) == 0
    @test sort.(collect.(findnz(t))) == ([1, 2, 4, 5], [1,1,1,7])
end

@testset "permutedims" begin
    t = strand(Float64, 5, 5, 5, 0.5)
    @test ndims(t) == 3
    @test permutedims(t, (2,1,3)) isa SparseTensor
    AT = Array(t)
    @test permutedims(t, (2,1,3)) == permutedims(AT, (2,1,3))
    @test t == AT
end

@testset "zero" begin
    # LongLongUInt
    t = randn(2, 2)
    t2 = SparseTensor{Float64, LongLongUInt{5}}(t)
    @test t2 isa SparseTensor{Float64, LongLongUInt{5}}
    @test t ≈ t2
end

@testset "sort and permute" begin
    @test bpermute(0b1100, [2,4,3,1]) === 0b0110
    @test bpermute(0b1100, [4,2,3,1]) === 0b0101
    @test bpermute(bit"1100", [2,4,3,1]) === bit"0110"
    @test bpermute(bit"1100", [4,2,3,1]) === bit"0101"
    @test sort(BitStr64{5}[1,8,4,2,9]) == sort([1,8,4,2,9])
end