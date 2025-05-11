############ BitBasis
@generated function ibcat(bits::NTuple{N, BitStr{M,T} where M}) where {N,T}
    total_bits = sum_length(bits.parameters...)
    quote
        val, len = T(0), 0
        @nexprs $N k->(val += buffer(bits[k]) << len; len += length(bits[k]))
        return BitStr{$total_bits,T}(val)
    end
end
sum_length(::Type{DitStr{D, N, T}}, dits...) where {D, N, T} = N + sum_length(dits...)
sum_length(::Type{DitStr{D, N, T}}) where {D, N, T} = N
allsame(x::T, mask::T) where T<:Integer = allone(x, mask) || !anyone(x, mask)

######################### index manipulation
function count_legs(ixs...)
    lc = Dict{eltype(ixs[1]),Int}()
    for l in Iterators.flatten(ixs)
        lc[l] = get(lc, l, 0) + 1
    end
    return lc
end

"""
return positions of dangling legs.
"""
function dangling_nleg_labels(ixs, iy, lc=count_legs(ixs..., iy))
    _dnlegs.(ixs, Ref(lc)), _dnlegs(iy, lc)
end
function _dnlegs(ix, lc)
    (unique(filter(iix->count(==(iix), ix)==lc[iix], [ix...]))...,)
end

function dumplicated_legs(ix)
    labels = OMEinsum.tunique(ix)
    findall(l->count(==(l), ix)>1, labels)
end

function allin(x, y)
    all(xi->xi ∈ y, x)
end
