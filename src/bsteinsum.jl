function cleanup_dangling_nlegs(ixs::Vector{Vector{LT}}, xs, iy::Vector{LT}) where LT
    lc = count_legs(ixs..., iy)
    danglegsin, danglegsout = dangling_nleg_labels(ixs, iy, lc)
    newxs = Any[xs...]
    newixs = [ixs...]
    newiy = iy
    for i = 1:length(xs)
        ix, dlx = newixs[i], danglegsin[i]
        if !isempty(dlx)
            newxs[i] = multidropsum(xs[i], dims=[findall(==(l), ix) for l in dlx])
            newixs[i] = filter(l->l∉dlx, [ix...])
        end
    end
    if !isempty(danglegsout)
        newiy = [l for l in newiy if l ∉ danglegsout]
    end
    return newixs, newxs, newiy
end

function cleanup_duplicated_legs(ixs::Vector{Vector{LT}}, xs, iy::Vector{LT}) where LT
    newxs = Any[xs...]
    newixs = collect(ixs)
    for i in 1:length(xs)
        ix = ixs[i]
        if !allunique(ix)
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
    @eval function OMEinsum.einsum(code::$ET, xs::NTuple{NT, BinarySparseTensor}, size_dict::Dict) where {NT}
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
        return einsum(EinCode([newnewiy], iy), (res,), size_dict)
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
    ymask = flip(Ti(0), bmask(Ti, 1:ndim))
    for red in reds
        ymask = unsetbit(ymask, bmask(Ti,red[2:end]...))
    end
    return ymask
end

function unsetbit(x::T, mask::T) where T<:Integer
    msk = ~T(0) ⊻ mask
    x & msk
end

function _ymask_from_trs(::Type{Ti}, ndim::Int, reds) where Ti
    ymask = flip(Ti(0), bmask(Ti, 1:ndim))
    for red in reds
        ymask = unsetbit(ymask, bmask(Ti, red...))
    end
    return ymask
end

function reduce_indices(t::BinarySparseTensor{Tv,Ti}, reds::Vector{Vector{LT}}) where {Tv,Ti,LT}
    inds = Ti[]
    vals = Tv[]
    ymask = _ymask_from_reds(Ti, ndims(t), reds)
    bits = baddrs(ymask)
    red_masks = [bmask(Ti, red...) for red in reds]
    for (ind, val) in zip(t.data.nzind, t.data.nzval)
        b = ind-1
        if all(red->allsame(b, red), red_masks)
            b = readbit(b, bits...)
            push!(inds, b+1)
            push!(vals, val)
        end
    end
    order = sortperm(inds)
    return BinarySparseTensor(SparseVector(1<<length(bits), inds[order], vals[order]))
end

function copy_indices(t::BinarySparseTensor{Tv,Ti}, targets::Vector{Vector{LT}}) where {Tv,Ti,LT}
    isempty(targets) && return t
    inds = Ti[]
    vals = Tv[]
    nbits = sum(length, targets)
    for (ind, val) in zip(t.data.nzind, t.data.nzval)
        b = ind-1
        b = copybits(b, targets)
        push!(inds, b+1)
        push!(vals, val)
    end
    order = sortperm(inds)
    return BinarySparseTensor(SparseVector(1<<nbits, inds[order], vals[order]))
end

function copybits(b::Ti, targets::Vector{Vector{LT}}) where {Ti,LT}
    res = Ti(0)
    for (i,t) in enumerate(targets)
        for it in t
            res |= readbit(b, i)<<(it-1)
        end
    end
    return res
end

function trace_indices(t::BinarySparseTensor{Tv,Ti}; dims::Vector{Vector{LT}}) where {Tv,Ti,LT}
    ymask = _ymask_from_trs(Ti, ndims(t), dims)
    bits = baddrs(ymask)
    red_masks = [bmask(Ti, red...) for red in dims]
    NO = length(bits)
    sv = SparseVector(1<<NO, Ti[], Tv[])
    for (ind, val) in zip(t.data.nzind, t.data.nzval)
        b = ind-1
        if all(red->allsame(b, red), red_masks)
            b = isempty(bits) ? zero(b) : readbit(b, bits...)
            sv[b+1] += val
        end
    end
    return BinarySparseTensor(sv)
end

Base._sum(f, t::BinarySparseTensor, ::Colon) = Base._sum(f, t.data, Colon())
function _dropsum(f, t::BinarySparseTensor{Tv,Ti,N}, dims) where {Tv,Ti,N}
    remdims = (setdiff(1:N, dims)...,)
    Tf = typeof(f(Tv(0)))
    d = Dict{Ti,Tf}()
    for (ind, val) in zip(t.data.nzind, t.data.nzval)
        rd = isempty(remdims) ? zero(ind) : readbit(ind-1, remdims...)
        d[rd] = get(d, rd, Tf(0)) + f(val)
    end
    ks = collect(keys(d))
    order = sortperm(ks)
    vals = collect(values(d))[order]
    return bst(SparseVector(1<<length(remdims), ks[order].+1, vals))
end

_dropsum(f, t::BinarySparseTensor, dims::Colon) = Base._sum(f, t, dims)
_dropsum(f, t::AbstractArray, dims::Colon) = Base._sum(f, t, dims)
_dropsum(f, t::AbstractArray, dims) = dropdims(Base._sum(f, t, dims), dims=dims)
dropsum(t::AbstractArray; dims=:) = _dropsum(identity, t, dims)

function multidropsum(t::BinarySparseTensor; dims)
    all(d->length(d) == 1, dims) && return dropsum(t; dims=first.(dims))
    trace_indices(t; dims=dims)
end
