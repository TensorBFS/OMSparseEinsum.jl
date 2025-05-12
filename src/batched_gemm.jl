function OMEinsum.get_output_array(xs::NTuple{N, SparseTensor{Tv,Ti,M} where {Tv,M}}, size::NTuple, fillzero::Bool) where {N,Ti}
    return stzeros(promote_type(map(eltype,xs)...), Ti, Ti.(size)...)
end

# a method to compute the batched gemm of two sparse tensors
# out: the output tensor
# dima, dimb: the dimensions of the two tensors
# ni, nb: the inner and batch dimensions of the two tensors
# inda, indb: the nonzero indices of the two tensors, assumed to have sorted inner and batch indices
# vala, valb: the values of the two tensors
# tensor indices are sorted as: (inner, batch, outer)
function batched_gemm_loops!(out::SparseTensor{Tv,Ti,M}, Noa, Nob, Nb, inda, indb, vala, valb) where {Tv,Ti,M}
    la, lb = 1, 1
    while lb <= length(indb) && la <= length(inda)
        # get inner and batch indices as fa, fb, get outer indices as foa, fob
        fa = div(inda[la] - 1, Noa)
        fb = div(indb[lb] - 1, Nob)
        @inbounds while fa != fb
            if fa < fb
                la += 1
                la > length(inda) && return
                fa = div(inda[la] - 1, Noa)
            else
                lb += 1
                lb > length(indb) && return
                fb = div(indb[lb] - 1, Nob)
            end
        end
        # get number of valid a
        na = 0
        @inbounds while la+na <= length(inda) && div(inda[la+na] - 1, Noa) == fb
            na += 1
        end

        nb = 0
        @inbounds while lb+nb <= length(indb) && div(indb[lb+nb] - 1, Nob) == fa
            nb += 1
        end
        for ka=la:la+na-1, kb=lb:lb+nb-1
            ia, va = inda[ka] - 1, vala[ka]
            ib, vb = indb[kb] - 1, valb[kb]
            # output indices: (batch, outera, outerb)
            indout = (ia % Noa) + (ib % Nob) * Noa + (fb % Nb) * (Noa*Nob)  # get outer indices
            accumindex!(out.data, va*vb, indout+1)
        end
        la += na
        lb += nb
    end
end

# accumulate a value in a dictionary, adapted from Base.get!
function accumindex!(h::Dict{K,V}, v::V, key::K) where V where K
    index, sh = Base.ht_keyindex2_shorthash!(h, key)
    if index > 0  # key exists
        h.vals[index] += v
        return nothing
    end

    # key absent, set value
    age0 = h.age
    if h.age != age0
        index, sh = Base.ht_keyindex2_shorthash!(h, key)
    end
    if index > 0
        h.age += 1
        @inbounds h.keys[index] = key
        @inbounds h.vals[index] = v
    else
        @inbounds Base._setindex!(h, v, key, -index, sh)
    end
    return nothing
end

# indice are sorted: (inner, batch, outer)
function sparse_contract!(out::SparseTensor, ni::Int, nb::Int, a::SparseTensor{T1,Ti,M}, b::SparseTensor{T2,Ti,N}) where {T1,T2,N,M,Ti}
    _ia, _va = findnz(a)  # TODO: check if needs copy
    _ib, _vb = findnz(b)
    ia, va, ib, vb = collect(_ia), collect(_va), collect(_ib), collect(_vb)
    ordera = sortperm(ia; lt=(x, y) -> x < y); ia, va = ia[ordera], va[ordera]
    orderb = sortperm(ib; lt=(x, y) -> x < y); ib, vb = ib[orderb], vb[orderb]
    noa, nob = M - ni - nb, N - ni - nb
    batched_gemm_loops!(out, prod(size(a)[1:noa]), prod(size(b)[1:nob]), prod(size(a)[noa+1:noa+nb]), ia, ib, va, vb)
    return out
end

function batched_contract(ixs, iy, xs::NTuple{NT, SparseTensor}) where {NT}
    # tensor indices are sorted as: (inner, batch, outer)
    pa, pb, pout, Ni, Nb = analyse_batched_perm(ixs..., iy)
    a = permutedims(xs[1], pa)
    b = permutedims(xs[2], pb)

    out = OMEinsum.get_output_array(xs, (size(a)[1:ndims(a)-Ni-Nb]..., size(b)[1:ndims(b)-Ni]...), true)
    sparse_contract!(out, Ni, Nb, a, b)
    return permutedims(out, pout)
end

# sort the indices: (inner, batch, outer)
function analyse_batched_perm(iAs, iBs, iOuts)
    iABs = iAs ∩ iBs
    pres   = iABs ∩ iOuts
    broad  = setdiff((iAs ∩ iOuts) ∪ (iBs ∩ iOuts), pres)
    summed = setdiff(iABs, pres)

    iAps, iAbs, iAss = pres ∩ iAs, broad ∩ iAs, summed ∩ iAs
    iBps, iBbs, iBss = pres ∩ iBs, broad ∩ iBs, summed ∩ iBs

    pA   = indexpos.(Ref(iAs), vcat(iAbs, iAps, iAss))
    pB   = indexpos.(Ref(iBs), vcat(iBbs, iBps, iBss))
    iABs = vcat(iAbs, iBbs, iAps)
    pOut = indexpos.(Ref(iABs), iOuts)

    return pA, pB, pOut, length(iAss), length(iAps)
end
indexpos(ix, item) = findfirst(==(item), ix)