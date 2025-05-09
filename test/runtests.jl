using SparseTN
using Test

@testset "BinarySparseTensor" begin
    include("BinarySparseTensor.jl")
end

@testset "bsteinsum" begin
    include("bsteinsum.jl")
end
