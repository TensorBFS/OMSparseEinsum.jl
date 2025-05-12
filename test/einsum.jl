using OMEinsum, SparseTN, BitBasis
using SparseTN: cleanup_duplicated_legs, cleanup_dangling_nlegs, dropsum, sparse_contract!
using Test
using SparseArrays

@testset "clean up tensors" begin
    ta = strand(Float64, Int, 4, 1.0)
    tb = strand(Float64, Int, 4, 1.0)
    ixs = [[3,4,5,6], [1,2,3,4]]
    iy = [1]
    newixs, newxs, newiy = SparseTN.cleanup_dangling_nlegs(ixs, [ta, tb], iy)
    @test newixs == [[3,4], [1,3,4]]
    @test newxs[1] ≈ dropsum(ta, dims=(3,4))
    @test newxs[2] ≈ dropsum(tb, dims=(2,))
    @test newiy == [1]

    # the output has dangling legs
    ixs = [[3,4,5,6], [1,2,3,4]]
    iy = [1, 8]
    newixs, newxs, newiy = SparseTN.cleanup_dangling_nlegs(ixs, [ta, tb], iy)
    @test newixs == [[3,4], [1,3,4]]
    @test newxs[1] ≈ dropsum(ta, dims=(3,4))
    @test newxs[2] ≈ dropsum(tb, dims=(2,))
    @test newiy == [1]
end

@testset "sparse contract" begin
    ta = strand(Float64, Int, 2,2,2,2, 0.5)
    tb = strand(Float64, Int, 2,2,2,2, 0.5)
    TA, TB = Array(ta), Array(tb)
    out = OMEinsum.get_output_array((ta,tb), (fill(2, 4)...,), true)
    @test sum(sparse_contract!(out, 2, 0, ta, tb)) ≈ sum(ein"lkji,nmji->lknm"(TA, TB))
    out = OMEinsum.get_output_array((ta,tb), (fill(2, 4)...,), true)
    @test sparse_contract!(out, 2, 0, ta, tb) ≈ ein"lkji,nmji->lknm"(TA, TB)

    # batched
    ta = strand(Float64, Int, 2,2,2,2,2, 0.5)
    tb = strand(Float64, Int, 2,2,2,2,2, 0.5)
    TA, TB = Array(ta), Array(tb)
    out = OMEinsum.get_output_array((ta,tb), (fill(2, 5)...,), true)
    @test sum(Array(sparse_contract!(out, 2, 1, ta, tb))) ≈ sum(ein"lkbji,nmbji->lknmb"(TA, TB))
    out = OMEinsum.get_output_array((ta,tb), (fill(2, 5)...,), true)
    @test Array(sparse_contract!(out, 2, 1, ta, tb)) ≈ ein"lkbji,nmbji->lknmb"(TA, TB)
end

@testset "einsum batched contract" begin
    perms = SparseTN.analyse_batched_perm(('b','j','c','a','e'), ('k','d','c','a','e'), ('b','j','k','d','c'))
    @test perms == ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], (1, 2, 3, 4, 5), 2, 1)

    sv = SparseVector([1.0,0,0,1,1,0,0,0])
    t1 = SparseTensor(sv, (2,2,2))
    t2 = SparseTensor(sv, (2,2,2))
    T1 = Array(t1)
    T2 = Array(t2)
    @test ein"ijk,jkl->il"(t1,t2) ≈ ein"ijk,jkl->il"(T1,T2)

    ta = strand(Float64, Int, 2, 2, 0.5)
    tb = strand(Float64, Int, 2, 2, 0.5)
    TA, TB = Array(ta), Array(tb)
    @test ein"ij,jk->ik"(ta,tb) ≈ ein"ij,jk->ik"(TA,TB)
    @test ta ≈ TA
    @test tb ≈ TB

    # with batch
    ta = strand(Float64, Int, 2, 2, 2, 2, 2, 2, 2, 0.5)
    tb = strand(Float64, Int, 2, 2, 2, 2, 2, 2, 0.5)
    TA, TB = Array(ta), Array(tb)
    code = ein"ijklmbc,ijbxcy->bcmlxky"
    res = code(ta, tb)
    @test res isa SparseTensor
    @test sum(res) ≈ sum(code(TA, TB))
    @test Array(res) ≈ code(TA, TB)
end

@testset "sum, ptrace and permute" begin
    ta = strand(Float64, Int, 2, 2, 2, 2, 2, 2, 2, 0.7)
    TA = Array(ta)
    res = SparseTN.dropsum(ta, dims=(2,4))
    @test res isa SparseTensor
    @test ndims(res) == 5
    @test Array(res) ≈ SparseTN.dropsum(TA, dims=(2,4))
    @test SparseTN.dropsum(ta) ≈ SparseTN.dropsum(TA)
    @test sum(ta) ≈ sum(TA)
    # sum
    res = ein"ijklbca->"(ta)
    @test res isa SparseTensor
    @test Array(res) ≈ ein"ijklbca->"(TA)
    res = ein"ijklbca->i"(ta)
    @test res isa SparseTensor
    @test Array(res) ≈ ein"ijklbca->i"(TA)

    # trace
    tb = strand(Float64, Int, 2, 2, 1.0)
    TB = Array(tb)
    res = ein"ii->"(tb)
    @test res isa SparseTensor
    @test Array(res) ≈ ein"ii->"(TB)

    # reduction
    code = ein"ijjlbbb->ijlb"
    res = code(ta)
    @test res isa SparseTensor
    @test Array(res) ≈ code(TA)
    code = ein"ijjlbbb->ljbi"
    res = code(ta)
    @test res isa SparseTensor
    @test Array(res) ≈ code(TA)

    # ptrace
    res = ein"ijjlbca->ailcb"(ta)
    @test res isa SparseTensor
    @test Array(res) |> sum ≈ ein"ijjlbca->ailcb"(TA) |> sum
    @test Array(res) ≈ ein"ijjlbca->ailcb"(TA)

    # permute
    res = ein"ijklbca->abcijkl"(ta)
    @test res isa SparseTensor
    @test Array(res) ≈ ein"ijklbca->abcijkl"(TA)
end

@testset "clean up tensors" begin
    ta = strand(Float64, Int, 1, 1.0)
    TA = Array(ta)
    # first reduce indices
    for code in [ein"i->iii", ein"i->jj", ein"k->kkj"]
        # TODO: add the rest tests
        if isempty(setdiff(OMEinsum.getiyv(code), union(OMEinsum.getixsv(code)...)))
            res = code(ta, size_info=Dict('j'=>2))
            @show code
            @test res isa SparseTensor
            @test res ≈ code(TA, size_info=Dict('j'=>2))
        end
    end
    ta = strand(Float64, Int, ntuple(i->2, 7)..., 0.5)
    TA = Array(ta)
    for code in [ein"iiiiiii->iiiiii", ein"iikjjjj->ikj", ein"iikjjjl->ikj"]
        @info code
        res = code(ta)
        @test res isa SparseTensor
        @test res ≈ code(TA)
    end
end

@testset "clean up" begin
    @test SparseTN.allsame(bit"000110", bmask(BitStr64{6}, 2,3))
    @test !SparseTN.allsame(bit"000110", bmask(BitStr64{6}, 2,4))
    @test SparseTN._get_reductions([1,2,2,4,3,1,5], [1,2,3,4,5]) == [[1,6], [2,3]]
    @test SparseTN.uniquelabels(ein"ijk,jkl->oo") == ['i', 'j', 'k', 'l', 'o']
end

@testset "count legs" begin
    @test SparseTN.count_legs((1,2), (2,3), (1,3)) == Dict(1=>2,2=>2,3=>2)
    @test SparseTN.dangling_nleg_labels(((1,1,2), (2,3)), (5,7), SparseTN.count_legs((1,1,2), (2,3), (5,7))) == (((1,), (3,)), (5,7))
end

@testset "unsetbit, copyidx" begin
    @test SparseTN.copyidx(0b110, [[1,2], [5,4], [3]], (1,2,4,8,16)) == 0b11100
end

@testset "binary with copy indices" begin
    sv = SparseVector([1.0,0,0,1,1,0,0,0])
    t1 = SparseTensor(sv, (2,2,2))
    t2 = SparseTensor(sv, (2,2,2))
    T1 = Array(t1)
    T2 = Array(t2)
    @test ein"ijk,jkl->ill"(t1,t2) ≈ ein"ijk,jkl->ill"(T1,T2)
end

@testset "longlong uint" begin
    T1 = rand(2, 2, 2)
    T1[T1 .< 0.5] .= 0
    T2 = rand(2, 2, 2)
    T2[T2 .< 0.5] .= 0

    t1 = SparseTensor{Float64, LongLongUInt{5}}(T1)
    t2 = SparseTensor{Float64, LongLongUInt{5}}(T2)
    @test ein"ijk,jkl->ill"(t1,t2) ≈ ein"ijk,jkl->ill"(T1,T2)
end