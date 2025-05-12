struct SparseTensor{Tv,Ti<:Integer,N} <: AbstractSparseArray{Tv, Ti, N}
    size::NTuple{N, Ti}
    strides::NTuple{N, Ti}
    data:: Dict{Ti, Tv}
end
function SparseTensor(data::SparseVector{Tv,Ti}, size::NTuple{N,Ti}) where {Tv,Ti,N}
    len = prod(size)
    @assert length(data) == len "data length should be $(len), got $(length(data))"
    d = Dict{Ti, Tv}()
    for (k, v) in zip(data.nzind, data.nzval)
        d[k] = v
    end
    strides = _size2strides(Ti, size)
    return SparseTensor{Tv, Ti, N}(size, strides, d)
end
_size2strides(::Type{Ti}, size::NTuple{N,Ti}) where {N,Ti} = ntuple(i->prod(size[1:i-1]; init=one(Ti)), N)
function SparseTensor(A::AbstractArray)
    SparseTensor(SparseVector(vec(A)), size(A))
end
function SparseTensor{Tv, Ti}(A::AbstractArray) where {Tv, Ti}
    d = Dict{Ti, Tv}()
    for i in 1:length(A)
        if A[i] != zero(Tv)
            d[Ti(i)] = A[i]
        end
    end
    return SparseTensor{Tv, Ti}(Ti.(size(A)), _size2strides(Ti, Ti.(size(A))), d)
end

function Base.getindex(t::SparseTensor{T,Ti,N}, index::Integer...) where {T,Ti,N}
    @boundscheck begin
        @assert length(index) == N || length(index) == 1 "index should have length $N, got $(length(index))"
        all(i -> 1<=index[i]<=size(t, i), 1:length(index)) || throw(BoundsError(t, index))
    end
    idx = cartesian2linear(t, index)
    return get(t.data, idx, zero(T))
end
function Base.getindex(t::SparseTensor{T,Ti,N}, index::Integer) where {T,Ti,N}
    @boundscheck 1<=index<=length(t) || throw(BoundsError(t, index))
    return get(t.data, index, zero(T))
end

cartesian2linear(st::SparseTensor, index) = sum(i->st.strides[i]*(index[i]-1), 1:length(index); init=1)
@generated function linear2cartesian(st::SparseTensor{Tv, Ti, N}, index::Integer) where {Tv, Ti, N}
    quote
        index = index - 1  # Convert to 0-based indexing for calculation
        @nexprs $N i -> begin
            idx_i, index = divrem(index, st.strides[$N - i + 1])
            idx_i = idx_i + 1
        end
        $(Expr(:tuple, [Symbol(:idx_, N-i+1) for i in 1:N]...))
    end
end

function Base.size(t::SparseTensor{T,Ti,N}, i::Int) where {T,Ti,N}
    i<=N ? t.size[i] : 1
end
Base.size(t::SparseTensor{T,Ti,N}) where {T,Ti,N} = t.size

Base.@propagate_inbounds function Base.setindex!(t::SparseTensor{T,Ti,N}, val, index::Integer...) where {T, Ti, N}
    return t.data[cartesian2linear(t, index)] = val
end
Base.@propagate_inbounds function Base.setindex!(t::SparseTensor{T,Ti,N}, val, index::Integer) where {T, Ti, N}
    return t.data[index] = val
end


SparseArrays.nnz(t::SparseTensor) = length(t.data)
function SparseArrays.findnz(t::SparseTensor{Tv,Ti,N}) where {Tv,Ti,N}
    return keys(t.data), values(t.data)
end
# used when converting to a dense array
SparseArrays.nonzeroinds(t::SparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = collect(keys(t.data))
SparseArrays.nonzeros(t::SparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = collect(values(t.data))

Base.show(io::IO, ::MIME"text/plain", t::SparseTensor) = Base.show(io, t)
function Base.show(io::IOContext, t::SparseTensor{T,Ti,1}) where {T,Ti}
    invoke(show, Tuple{IO,SparseTensor}, io, t)
end
function Base.show(io::IO, t::SparseTensor{T,Ti,N}) where {T,Ti,N}
    NNZ = length(t.data)
    println(io, "$(summary(t)) with $(nnz(t)) stored entries:")
    for (i, (nzi, nzv)) in enumerate(zip(findnz(t)...))
        print(io, "  $(linear2cartesian(t, nzi)) => $nzv")
        i != NNZ && println(io)
    end
end

stzeros(::Type{Tv}, ::Type{Ti}, size::Vararg{Ti, N}) where {Tv,Ti<:Integer,N} = SparseTensor{Tv,Ti,N}(size, _size2strides(Ti, size), Dict{Ti, Tv}())
Base.zero(t::SparseTensor{Tv,Ti,N}) where {Tv,Ti,N} = SparseTensor{Tv,Ti,N}(t.size, t.strides, Dict{Ti, Tv}())
Base.copy(t::SparseTensor{Tv,Ti,N}) where {Tv,Ti,N} = SparseTensor{Tv,Ti,N}(t.size, t.strides, copy(t.data))

# random sparse tensor
strand(::Type{Tv}, ::Type{Ti}, args::Real...) where {Tv, Ti} = SparseTensor(SparseVector{Tv, Ti}(sprand(Tv, prod(args[1:end-1]), args[end])), args[1:end-1])

function Base.permutedims!(dest::SparseTensor{Tv,Ti,N}, src::SparseTensor{Tv,Ti,N}, dims::NTuple{N,Int}) where {Tv,Ti,N}
    for (k, v) in src.data
        dest.data[bpermute(k, dims, src, dest)] = v
    end
    return dest
end
# permute bits in an integer
function bpermute(b::T, order::NTuple{N,Integer}, src::SparseTensor, dest::SparseTensor) where {T<:Integer,N}
    res = zero(b)
    ci = linear2cartesian(src, b)
    for (i, o) in enumerate(order)
        res += (ci[o] - 1) * dest.strides[i]
    end
    return res + 1
end

function Base.permutedims(src::SparseTensor{Tv,Ti,N}, dims) where {Tv,Ti,N}
    @assert length(dims) == N "dims should have length $N, got $(length(dims))"
    issorted(dims) && return src
    dest = zero(src)
    return Base.permutedims!(dest, src, (dims...,))
end