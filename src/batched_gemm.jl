# OMEinsum.asarray(x::Number, arr::BinarySparseTensor) = BinarySparseTensor(OMEinsum.asarray(x, arr.data))

function OMEinsum.get_output_array(xs::NTuple{N, BinarySparseTensor{Tv,Ti,M} where {Tv,M}}, size, fillzero::Bool) where {N,Ti}
    return bst_zeros(promote_type(map(eltype,xs)...), Ti, length(size))
end

# a method to compute the batched gemm of two sparse tensors
# g: a function that operates on the indices
# inda, indb: the nonzero indices of the two tensors, assumed to have sorted inner and batch indices
# ni, nb: the inner and batch dimensions of the two tensors
function chasing_game(g, dima::Int, dimb::Int, ni::Int, nb::Int, inda, indb)
    offseta = dima - nb - ni
    offsetb = dimb - nb - ni
    la, lb = 1, 1
    while lb <= length(indb) && la <= length(inda)
        fa = (inda[la] - 1) >> offseta
        fb = (indb[lb] - 1) >> offsetb
        @inbounds while fa != fb
            if fa < fb
                la += 1
                la > length(inda) && return
                fa = (inda[la] - 1) >> offseta
            else
                lb += 1
                lb > length(indb) && return
                fb = (indb[lb] - 1) >> offsetb
            end
        end
        # get number of valid a
        na = 0
        @inbounds while la+na <= length(inda) && (inda[la+na] - 1) >> offseta == fb
            na += 1
        end

        nb = 0
        @inbounds while lb+nb <= length(indb) && (indb[lb+nb] - 1) >> offsetb == fa
            nb += 1
        end
        for ka=la:la+na-1, kb=lb:lb+nb-1
            g(ka, kb)
        end
        la += na
        lb += nb
    end
end

# indice are sorted: (inner, batch, outer)
function sparse_contract!(out::BinarySparseTensor, ni::Int, nb::Int, a::BinarySparseTensor{T1,Ti,M}, b::BinarySparseTensor{T2,Ti,N}) where {T1,T2,N,M,Ti}
    noa, nob = M-ni-nb, N-ni-nb
    outermaska = bmask(1:noa)
    outermaskb = bmask(1:nob)
    batchmask = bmask(1:nb)

    ia, va = copy.(findnz(a))
    ib, vb = copy.(findnz(b))
    chasing_game(M, N, ni, nb, ia, ib) do la, lb
        inda, vala = (ia[la] - 1), va[la]
        indb, valb = (ib[lb] - 1), vb[lb]
        # output indices: (batch, outera, outerb)
        indout = (inda & outermaska) | ((indb & outermaskb) << noa) | (((indb >> nob) & batchmask) << (noa+nob))  # get outer indices
        @inbounds out[indout+1] += vala*valb
    end
    return out
end

function batched_contract(ixs, iy, xs::NTuple{NT, BinarySparseTensor}) where {NT}
    a, b = xs
    pa, pb, pout, Ni, Nb = analyse_batched_perm(ixs..., iy)
    a = permutedims(a, pa)
    b = permutedims(b, pb)

    out = OMEinsum.get_output_array(xs, (fill(2, ndims(a)+ndims(b)-2Ni-Nb)...,), true)
    sparse_contract!(out, Ni, Nb, a, b)
    permutedims(out, pout)
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