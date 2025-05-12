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
    @assert length(iy_mid) == ndims(x)
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

function reduce_indices(t::SparseTensor{Tv,Ti,N}, reds::Vector{Vector{LT}}) where {Tv,Ti,N,LT}
    @assert all(red -> allequal(r -> size(t, r), red), reds) "reduced dimensions are not the same: given $reds, got size: $(size(t))"
    taken_dims = setdiff(1:ndims(t), vcat([red[2:end] for red in reds]...))
    sz = ntuple(i->size(t, taken_dims[i]), length(taken_dims))
    out = stzeros(Tv, Ti, sz...)
    taken_strides = zeros(Ti, N)
    taken_strides[taken_dims] .= out.strides
    return _reduce_indices!(out, t, reds, (taken_strides...,))
end

function _reduce_indices!(out::SparseTensor, t::SparseTensor{Tv,Ti,N}, reds, taken_strides) where {Tv,Ti,N}
    for (ind, val) in t.data
        ci = linear2cartesian(t, ind)
        if all(red -> allequal(k -> ci[k], red), reds)
            accumindex!(out.data, val, cartesian2linear(Ti, taken_strides, ci))
        end
    end
    return out
end
# masked locations are all 1s or 0s
allsame(x::T, mask::T) where T<:Integer = allone(x, mask) || !anyone(x, mask)

# copy indices from 1:ndims(t) to targets, return a new SparseTensor
# targets is a vector of vectors, the length is the number of dimensions of the input tensors
function copy_indices(t::SparseTensor{Tv,Ti}, targets::Vector{Vector{LT}}) where {Tv,Ti,LT}
    @assert ndims(t) == length(targets)
    isempty(targets) && return t

    # get size of the output tensor
    vtar = vcat(targets...)
    szv = zeros(Ti, length(vtar))
    for (i, js) in enumerate(targets)
        for j in js
            szv[j] = size(t, i)
        end
    end
    target_strides = _size2strides(Ti, szv)
    inds, vals = _copy_indices(t, targets, collect(Ti, target_strides))
    return SparseTensor{Tv, Ti, length(szv)}((szv...,), target_strides, Dict(zip(inds, vals)))
end

# copy indices from t to targets
function _copy_indices(t::SparseTensor{Tv,Ti, N}, targets::Vector{Vector{LT}}, target_strides::Vector{Ti}) where {Tv,Ti,LT,N}
    # copy indices
    inds = Vector{Ti}(undef, length(t.data))
    vals = Vector{Tv}(undef, length(t.data))
    for (k, (ind, val)) in enumerate(t.data)
        newind = copyidx(ind, targets, t.strides, target_strides)
        inds[k] = newind
        vals[k] = val
    end
    return inds, vals
end

function copyidx(ind::Ti, targets, strides_source::NTuple{N1,Ti}, strides_target) where {Ti,N1}
    ci = linear2cartesian(Ti, strides_source, ind)
    res = one(Ti)
    for (i,t) in enumerate(targets)
        for it in t
            res += (ci[i] - 1) * strides_target[it]
        end
    end
    return res
end

function trace_indices(t::SparseTensor{Tv,Ti}; dims::Vector{Vector{LT}}) where {Tv,Ti,LT}
    remaining_dims = setdiff(1:ndims(t), vcat(dims...))
    sz = ntuple(i->size(t, remaining_dims[i]), length(remaining_dims))
    out = stzeros(Tv, Ti, sz...)
    rem_strides = zeros(Ti, ndims(t))
    rem_strides[remaining_dims] .= out.strides
    return _trace_indices!(out, t, dims, (rem_strides...,))
end

function _trace_indices!(out::SparseTensor{Tv,Ti,N1}, t::SparseTensor{Tv,Ti,N2}, trace_dims::Vector{Vector{LT}}, rem_strides::NTuple{N2,Ti}) where {Tv,Ti,N1,N2,LT}
    for (ind, val) in t.data
        ci = linear2cartesian(t, ind)
        if all(red->allequal(k -> ci[k], red), trace_dims)
            accumindex!(out.data, val, cartesian2linear(Ti, rem_strides, ci))
        end
    end
    return out
end

Base._sum(f, t::SparseTensor, ::Colon) = Base._sum(f, values(t.data), Colon())
function _remsum(f, t::SparseTensor{Tv,Ti,N}, remdims::NTuple{NR}, newstrides::NTuple{NR,Ti}) where {Tv,Ti,N,NR}
    Tf = typeof(f(zero(Tv)))
    d = Dict{Ti,Tf}()
    for (ind, val) in t.data
        ci = linear2cartesian(t, ind)
        rd = isempty(remdims) ? one(ind) : cartesian2linear(Ti, newstrides, ntuple(i->ci[remdims[i]], NR))
        accumindex!(d, f(val), rd)
    end
    sz = map(i->size(t, i), remdims)
    return SparseTensor{Tf, Ti, NR}(sz, _size2strides(Ti, sz), d)
end
dropsum(f, t::SparseTensor{Tv,Ti,N}; dims=:) where {Tv,Ti,N} = dims == Colon() ? Base._sum(f, t, Colon()) : _remsum(f, t, (setdiff(1:ndims(t), dims)...,), _size2strides(Ti, map(i->size(t, i), setdiff(1:ndims(t), dims))))
dropsum(f, t::AbstractArray; dims=:) = dims == Colon() ? Base._sum(f, t, Colon()) : dropdims(Base._sum(f, t, dims), dims=dims)
dropsum(t::AbstractArray; dims=:) = dropsum(identity, t; dims)

function multidropsum(t::SparseTensor; dims)
    all(d->length(d) == 1, dims) && return dropsum(t; dims=first.(dims))
    trace_indices(t; dims=dims)
end
