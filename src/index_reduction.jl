function OMEinsum.unary_einsum!(::OMEinsum.Sum, ix, iy, x::BinarySparseTensor, y::BinarySparseTensor, sx, sy)
    dims = (findall(i -> i ∉ iy, ix)...,)
    ix1f = filter!(i -> i in iy, collect(ix))
    perm = map(i -> findfirst(==(i), ix1f), iy)
    res = dropsum(xs[1], dims=dims)
    perm == iy ? res : permutedims(res, perm)
end

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

function cleanup_dumplicated_legs(ixs::Vector{Vector{LT}}, xs, iy::Vector{LT}) where LT
    newxs = Any[xs...]
    newixs = collect(ixs)
    for i in 1:length(xs)
        ix = ixs[i]
        if !allunique(ix)
            newix = unique(ix)
            newxs[i] = impl_reduction(EinCode([ix], newix), (xs[i],), OMEinsum.get_size_dict([ix], (xs[i],)))
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
            ix, x = ixs[1], xs[1]
            return einsum_unary(ix, iy, x, size_dict)
        end
        # dangling legs or multi-legs
        ycode = empty(iy)
        newixs, newxs, newiy = cleanup_dangling_nlegs(ixs, xs, iy)
        @show newixs, newiy
        newiy!=iy && pushfirst!(ycode, EinCode((newiy,), iy))
        newixs, newxs, newnewiy = cleanup_dumplicated_legs(newixs, newxs, newiy)
        newiy!=newnewiy && pushfirst!(ycode, EinCode((newnewiy,), newiy))
        @show ycode
        #newixs == ixs && newnewiy == iy && throw(ArgumentError("Einsum not implemented for $code"))
        # dangling merge multi-legs to one leg
        res = einsum(EinCode(newixs, newnewiy), (newxs...,), size_dict)
        for code in ycode
            # TODO: broadcast and duplicate
            res = einsum(code, (res,), size_dict)
        end
        return res
    end
end

function einsum_unary(ix, iy, x, size_dict)
    Nx = length(ix)
    Ny = length(iy)
    # the first rule with the higher the priority
    if Ny == 0 && Nx == 2 && ix[1] == ix[2]    # trace
        return impl_trace(ix, iy, x, size_dict)
    elseif allunique(iy)
        @info iy
        if ix == iy
            return x
        elseif allunique(ix)
            if Nx == Ny
                if all(i -> i in iy, ix)
                    return impl_permutedims(ix, iy, x, size_dict)
                else  # e.g. (abcd->bcde)
                    throw(ArgumentError("Einsum not implemented for $ix -> $iy"))
                end
            else
                if all(i -> i in ix, iy)
                    return impl_sum(ix, iy, x, size_dict)
                elseif all(i -> i in iy, ix)  # e.g. ij->ijk
                    return impl_repeat(ix, iy, x, size_dict)
                else  # e.g. ijkxc,ijkl
                    throw(ArgumentError("Einsum not implemented for $ix -> $iy"))
                end
            end
        else  # ix is not unique
            if all(i -> i in ix, iy) && all(i -> i in iy, ix)   # ijjj->ij
                return impl_diag(ix, iy, x, size_dict)
            else
                throw(ArgumentError("Einsum not implemented for $ix -> $iy"))
            end
        end
    else  # iy is not unique
        if allunique(ix) && all(x->x∈iy, ix)
            if all(y->y∈ix, iy)  # e.g. ij->ijjj
                return impl_repeat(ix, iy, x, size_dict)
            else  # e.g. ij->ijjl
                throw(ArgumentError("Einsum not implemented for $ix -> $iy"))
            end
        else
            throw(ArgumentError("Einsum not implemented for $ix -> $iy"))
        end
    end
end

function is_reduction(ix::Vector{LT}, iy::Vector{LT}) where LT
    isempty(ix) && return false  # avoid matching "->"
    (allunique(ix) || !allunique(iy)) && return false
    allin(iy, ix) && allin(ix, iy) || return false
    return true
end

function is_copy(ix::Vector{LT}, iy::Vector{LT}) where LT
    isempty(ix) && return false  # avoid matching "->"
    (!allunique(ix) || allunique(iy)) && return false
    allin(iy, ix) && allin(ix, iy) || return false
    return true
end

function is_broadcast(ix::Vector{LT}, iy::Vector{LT}) where LT
    isempty(ix) && return false  # avoid matching "->"
    !allunique(ix) && return false
    allin(ix, iy) || return false
    filter(iiy->(iiy ∈ ix), [iy...]) == [ix...] || return false
    return true
end

function _get_reductions(ix::Vector{LT}, iy::Vector{LT}) where LT
    reds = Vector{Int}[]
    for l in iy
        count(==(l), ix) > 1 && push!(reds, findall(==(l), ix))
    end
    return reds
end

function _get_traces(ix::Vector{LT}, iy::Vector{LT}) where LT
    reds = Vector{Int}[]
    for l in unique(ix)
        l ∉ iy && count(==(l), ix) > 1 && push!(reds, findall(==(l), ix))
    end
    return reds
end

function impl_reduction(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Reduction" ix => iy size(x)
    reduce_indices(x, _get_reductions(ix, iy))  
end

function impl_copy(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Copy" ix => iy size(x)
    copy_indices(x, map(l->(findall(==(l), iy)...,), ix))
end

function impl_broadcast(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Broadcast" ix => iy size(x)
    throw(ArgumentError("Not implemented"))
end

function impl_trace(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Trace" ix => iy size(x) newix
    res = trace_indices(x; dims=_get_traces(ix, iy))
    newix = filter(ix->ix ∈ iy, [ix...])
    return einsum(EinCode([newix], iy), (res,), size_dict)
end

function impl_sum(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Sum" ix => iy size(x) dims
    dims = (findall(i -> i ∉ iy, ix)...,)
    return dropsum(x, dims=dims)
end

function impl_permutedims(ix::Vector{LT}, iy::Vector{LT}, x::BinarySparseTensor, size_dict) where LT
    @debug "Permutedims" ix => iy size(x) perm
    perm = ntuple(i -> findfirst(==(iy[i]), ix)::Int, length(iy))
    return tensorpermute(x, perm)
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
    @show nbits
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
