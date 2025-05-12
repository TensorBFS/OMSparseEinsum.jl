using SparseTN
using Test

@testset "SparseTensor" begin
    include("SparseTensor.jl")
end

@testset "einsum" begin
    include("einsum.jl")
end
