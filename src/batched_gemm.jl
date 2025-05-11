# OMEinsum.asarray(x::Number, arr::BinarySparseTensor) = BinarySparseTensor(OMEinsum.asarray(x, arr.data))

function OMEinsum.get_output_array(xs::NTuple{N, BinarySparseTensor{Tv,Ti,M} where {Tv,M}}, size, fillzero::Bool) where {N,Ti}
    return bst_zeros(promote_type(map(eltype,xs)...), Ti, length(size))
end

# a method to compute the batched gemm of two sparse tensors
# innerbatch: a function that returns the inner and batch indices
# g: a function that operates on the indices
# inda, indb: the nonzero indices of the two tensors, assumed to have sorted inner and batch indices
function chasing_game(g, innerbatch, inda, indb)
    la, lb = 1, 1
    while lb <= length(indb) && la <= length(inda)
        @inbounds fa = innerbatch(inda[la])
        @inbounds fb = innerbatch(indb[lb])
        @inbounds while fa != fb
            if fa < fb
                la += 1
                la > length(inda) && return
                fa = innerbatch(inda[la])
            else
                lb += 1
                lb > length(indb) && return
                fb = innerbatch(indb[lb])
            end
        end
        # get number of valid a
        na = 0
        @inbounds while la+na <= length(inda) && innerbatch(inda[la+na]) == fb
            na += 1
        end

        nb = 0
        @inbounds while lb+nb <= length(indb) && innerbatch(indb[lb+nb]) == fa
            nb += 1
        end
        for ka=la:la+na-1, kb=lb:lb+nb-1
            g(ka, kb)
        end
        la += na
        lb += nb
    end
end

function get_inner(::Val{Ni}, x::BitStr{N,T}) where {N, Ni, T}
    BitStr{Ni,T}(x >> (N-Ni))
end

function get_batch(::Val{Ni}, ::Val{Nb}, x::BitStr{N,T}) where {N, Ni, Nb, T}
    BitStr{Nb,T}((x >> (N-Ni-Nb)) & bmask(1:Nb))
end

function get_inner_and_batch(::Val{Ni}, ::Val{Nb}, x::BitStr{N,T}) where {N, Nb, Ni, T}
    BitStr{Nb+Ni,T}(x >> (N-Nb-Ni))
end

function get_outer(::Val{Ni}, ::Val{Nb}, x::BitStr{N,T}) where {Ni,Nb,N,T}
    BitStr{N-Ni-Nb,T}(x & bmask(1:N-Ni-Nb))
end

function get_outer(ni::Val{Ni}, nb::Val{Nb}, xs::BitStr...) where {Ni,Nb}
    ibcat((get_outer.(ni, nb, xs)...,get_batch(ni, nb, xs[1])))
end

function sparse_contract(ni::Val{Ni}, nb::Val{Nb}, a::BinarySparseTensor{T1,Ti,M}, b::BinarySparseTensor{T2,Ti,N}) where {T1,T2,Ni,Nb,N,M,Ti}
    out = OMEinsum.get_output_array((a,b), (fill(2, M+N-2Ni-Nb)...,), true)
    ia, va = findnz(a)
    ib, vb = findnz(b)
    chasing_game(x->get_inner_and_batch(ni, nb, x), ia,ib) do la, lb
    # for (la, lb) in naive_chase(get_inner_and_batch.(ni, nb, ia), get_inner.(ni, nb, ib))
        inda, vala = ia[la], va[la]
        indb, valb = ib[lb], vb[lb]
        indout = get_outer(ni, nb, inda, indb)
        out[indout] += vala*valb
    end
    return out
end

function batched_contract(ixs, iy, xs::NTuple{NT, BinarySparseTensor}) where {NT}
    a, b = xs
    pa, pb, pout, Ni, Nb = analyse_batched_perm(ixs..., iy)
    A = copy(a)
    B = copy(b)
    a = permutedims(a, pa)
    b = permutedims(b, pb)
    out = sparse_contract(Val(Ni), Val(Nb), a, b)
    permutedims(out, pout)
end

# can be used in either static or dynamic invoke
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