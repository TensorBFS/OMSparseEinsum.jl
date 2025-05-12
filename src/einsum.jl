function cleanup_dangling_nlegs(ixs::Vector{Vector{LT}}, xs, iy::Vector{LT}) where LT
    lc = count_legs(ixs..., iy)
    danglegsin, danglegsout = dangling_nleg_labels(ixs, iy, lc)
    newxs = Any[xs...]
    newixs = [ixs...]
    newiy = iy
    for (i, x) in enumerate(xs)
        ix, dlx = newixs[i], danglegsin[i]
        if !isempty(dlx)
            newxs[i] = multidropsum(x, dims=[findall(==(l), ix) for l in dlx])
            newixs[i] = filter(l->l∉dlx, [ix...])
        end
    end
    if !isempty(danglegsout)
        newiy = [l for l in newiy if l ∉ danglegsout]
    end
    return newixs, newxs, newiy
end

# return positions of dangling legs.
function dangling_nleg_labels(ixs, iy, lc)
    _dnlegs.(ixs, Ref(lc)), _dnlegs(iy, lc)
end
function _dnlegs(ix, lc)
    (unique(filter(iix->count(==(iix), ix)==lc[iix], [ix...]))...,)
end
function count_legs(ixs...)
    lc = Dict{eltype(ixs[1]),Int}()
    for l in Iterators.flatten(ixs)
        accumindex!(lc, 1, l)
    end
    return lc
end

function cleanup_duplicated_legs(ixs::Vector{Vector{LT}}, xs, iy::Vector{LT}) where LT
    newxs = Any[xs...]
    newixs = collect(ixs)
    for (i, ix) in enumerate(ixs)
        if !allunique(ix)  # duplicated legs
            newix = unique(ix)
            newxs[i] = reduce_indices(xs[i], _get_reductions(ix, newix))
            newixs[i] = newix
        end
    end
    newiy = iy
    if !allunique(newiy)
        newiy = unique(newiy)
    end
    return newixs, newxs, newiy
end

for ET in [:StaticEinCode, :DynamicEinCode]
    @eval function OMEinsum.einsum(code::$ET, xs::NTuple{NT, SparseTensor}, size_dict::Dict) where {NT}
        ixs, iy = OMEinsum.getixsv(code), OMEinsum.getiyv(code)
        if length(ixs) == 1   # unary operations
            return einsum_unary(ixs[1], iy, xs[1])
        end
        # clean up dangling legs or multi-legs
        newixs, newxs, newiy = cleanup_dangling_nlegs(ixs, xs, iy)
        newixs, newxs, newnewiy = cleanup_duplicated_legs(newixs, newxs, newiy)

        # dangling merge multi-legs to one leg
        res = batched_contract(newixs, newnewiy, (newxs...,))

        # Duplicate (or diag)
        res = einsum(EinCode([newnewiy], iy), (res,), size_dict)
        return res
    end
end

function einsum_unary(ix, iy, x)
    if !isempty(setdiff(iy, ix))
        throw(ArgumentError("Einsum not implemented for unary operation with new output indices: $ix -> $iy"))
    end
    iy_mid = unique(iy)

    # Remove dimensions that can be traced out
    trace_dims = _get_ptraces(ix, iy_mid)
    if !isempty(trace_dims)
        x = trace_indices(x; dims=trace_dims)
        ix = filter(ix->ix ∈ iy_mid, [ix...])
    end

    # Reduce duplicated indices
    reduce_dims = _get_reductions(ix, iy_mid)
    if !isempty(reduce_dims)
        x = reduce_indices(x, reduce_dims)
        ix = unique(ix)
    end

    # Permute to the wanted order
    perm = map(item -> findfirst(==(item), ix), iy_mid)
    x = permutedims(x, perm)

    # Copy indices
    copy_indices(x, map(l->findall(==(l), iy), iy_mid))
end

function _get_reductions(ix::Vector{LT}, iy::Vector{LT}) where LT
    reds = Vector{Int}[]
    for l in iy
        count(==(l), ix) > 1 && push!(reds, findall(==(l), ix))
    end
    return reds
end

function _get_ptraces(ix::Vector{LT}, iy::Vector{LT}) where LT
    reds = Vector{Int}[]
    for l in unique(ix)
        l ∉ iy && push!(reds, findall(==(l), ix))
    end
    return reds
end

function _ymask_from_reds(::Type{Ti}, ndim::Int, reds) where Ti
    ymask = flip(zero(Ti), bmask(Ti, 1:ndim))
    for red in reds
        ymask = unsetbit(ymask, bmask(Ti,red[2:end]...))
    end
    return ymask
end

function unsetbit(x::T, mask::T) where T<:Integer
    msk = ~zero(T) ⊻ mask
    x & msk
end

function _ymask_from_trs(::Type{Ti}, ndim::Int, reds) where Ti
    ymask = flip(zero(Ti), bmask(Ti, 1:ndim))
    for red in reds
        ymask = unsetbit(ymask, bmask(Ti, red...))
    end
    return ymask
end

function reduce_indices(t::SparseTensor{Tv,Ti,N}, reds::Vector{Vector{LT}}) where {Tv,Ti,N,LT}
    inds = Ti[]
    vals = Tv[]
    ymask = _ymask_from_reds(Ti, N, reds)
    bits = baddrs(ymask)
    red_masks = [bmask(Ti, red...) for red in reds]
    for (ind, val) in t.data
        b = ind-1
        if all(red->allsame(b, red), red_masks)
            b = readbit(b, bits...)
            push!(inds, b+1)
            push!(vals, val)
        end
    end
    return SparseTensor{Tv, Ti, N-sum(x -> length(x) - 1, reds)}(Dict(zip(inds, vals)))
end
# masked locations are all 1s or 0s
allsame(x::T, mask::T) where T<:Integer = allone(x, mask) || !anyone(x, mask)

function copy_indices(t::SparseTensor{Tv,Ti}, targets::Vector{Vector{LT}}) where {Tv,Ti,LT}
    isempty(targets) && return t
    inds = Ti[]
    vals = Tv[]
    nbits = sum(length, targets)
    for (ind, val) in t.data
        b = ind-1
        b = copybits(b, targets)
        push!(inds, b+1)
        push!(vals, val)
    end
    return SparseTensor{Tv, Ti, nbits}(Dict(zip(inds, vals)))
end

function copybits(b::Ti, targets::Vector{Vector{LT}}) where {Ti,LT}
    res = zero(Ti)
    for (i,t) in enumerate(targets)
        for it in t
            res |= readbit(b, i)<<(it-1)
        end
    end
    return res
end

function trace_indices(t::SparseTensor{Tv,Ti}; dims::Vector{Vector{LT}}) where {Tv,Ti,LT}
    ymask = _ymask_from_trs(Ti, ndims(t), dims)
    bits = baddrs(ymask)
    red_masks = [bmask(Ti, red...) for red in dims]
    NO = length(bits)
    sv = stzeros(Tv, Ti, NO)
    for (ind, val) in t.data
        b = ind-1
        if all(red->allsame(b, red), red_masks)
            b = isempty(bits) ? zero(b) : readbit(b, bits...)
            accumindex!(sv.data, val, b+1)
        end
    end
    return sv
end

Base._sum(f, t::SparseTensor, ::Colon) = Base._sum(f, values(t.data), Colon())
function _remsum(f, t::SparseTensor{Tv,Ti,N}, remdims::NTuple{NR}) where {Tv,Ti,N,NR}
    Tf = typeof(f(zero(Tv)))
    d = Dict{Ti,Tf}()
    for (ind, val) in t.data
        rd = isempty(remdims) ? zero(ind) : readbit(ind-1, remdims...)
        accumindex!(d, f(val), rd + 1)
    end
    return SparseTensor{Tf, Ti, NR}(d)
end
dropsum(f, t::SparseTensor; dims=:) = dims == Colon() ? Base._sum(f, t, Colon()) : _remsum(f, t, (setdiff(1:ndims(t), dims)...,))
dropsum(f, t::AbstractArray; dims=:) = dims == Colon() ? Base._sum(f, t, Colon()) : dropdims(Base._sum(f, t, dims), dims=dims)
dropsum(t::AbstractArray; dims=:) = dropsum(identity, t; dims)

function multidropsum(t::SparseTensor; dims)
    all(d->length(d) == 1, dims) && return dropsum(t; dims=first.(dims))
    trace_indices(t; dims=dims)
end
