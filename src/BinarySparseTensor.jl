# binary sparse tensor
struct BinarySparseTensor{Tv,Ti<:Integer,N} <: AbstractSparseArray{Tv, Ti, N}
   data:: Dict{Ti, Tv}  # indices must be sorted!
end
function BinarySparseTensor(data::SparseVector{Tv,Ti}) where {Tv,Ti}
    N = log2i(length(data))
    length(data) != one(Ti) << N && throw(ArgumentError("data length should be 2^x, got $(length(data))"))
    d = Dict{Ti, Tv}()
    for (k, v) in zip(data.nzind, data.nzval)
        d[k] = v
    end
    return BinarySparseTensor{Tv, Ti, N}(d)
end
function BinarySparseTensor(A::AbstractArray)
    BinarySparseTensor(SparseVector(vec(A)))
end
function BinarySparseTensor{Tv, Ti}(A::AbstractArray) where {Tv, Ti}
    sv = SparseVector(vec(A))
    BinarySparseTensor(SparseVector{Tv, Ti}(Ti(length(A)), Ti.(sv.nzind), sv.nzval))
end

function Base.getindex(t::BinarySparseTensor{T,Ti,N}, index::BitStr{N}) where {T,Ti,N}
    idx = as_index(Ti, index)
    @boundscheck idx <= one(Ti) << N || throw(BoundsError(t, index))
    @inbounds return get(t.data, idx, zero(T))
end
function Base.getindex(t::BinarySparseTensor{T,Ti,N}, index::Integer...) where {T,Ti,N}
    idx = as_index(Ti, index)
    @boundscheck idx <= one(Ti) << N || throw(BoundsError(t, index))
    @inbounds return get(t.data, idx, zero(T))
end
as_index(::Type{Ti}, x::Integer) where {Ti} = Ti(x)
as_index(::Type{Ti}, x::BitStr) where {Ti} = Ti(buffer(x)+1)
@inline function as_index(::Type{Ti}, x::NTuple{N,<:Integer}) where {Ti,N}
    res = one(Ti)
    for i=1:N
        @inbounds res += Ti(x[i]-1)<<(i-1)
    end
    return res
end

function Base.size(t::BinarySparseTensor{T,Ti,N}, i::Int) where {T,Ti,N}
    @boundscheck i<=0 && throw(BoundsError(size(t), i))
    i<=N ? 2 : 1
end
Base.size(t::BinarySparseTensor{T,Ti,N}) where {T,Ti,N} = ntuple(i->2, N)

Base.@propagate_inbounds function Base.setindex!(t::BinarySparseTensor{T,Ti,N}, val, index) where {T, Ti, N}
    return t.data[as_index(Ti, index)] = val
end

SparseArrays.nnz(t::BinarySparseTensor) = length(t.data)
function SparseArrays.findnz(t::BinarySparseTensor{Tv,Ti,N}) where {Tv,Ti,N}
    return keys(t.data), values(t.data)
end
SparseArrays.nonzeroinds(t::BinarySparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = collect(keys(t.data))
SparseArrays.nonzeros(t::BinarySparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = collect(values(t.data))
# Base.Array(t::BinarySparseTensor{Tv,Ti,1}) where {Tv,Ti} = Base.Array(t.data)

Base.show(io::IO, ::MIME"text/plain", t::BinarySparseTensor) = Base.show(io, t)
function Base.show(io::IOContext, t::BinarySparseTensor{T,Ti,1}) where {T,Ti}
    invoke(show, Tuple{IO,BinarySparseTensor}, io, t)
end
function Base.show(io::IO, t::BinarySparseTensor{T,Ti,N}) where {T,Ti,N}
    NNZ = length(t.data)
    println(io, "$(summary(t)) with $(nnz(t)) stored entries:")
    for (i, (nzi, nzv)) in enumerate(zip(findnz(t)...))
        print(io, "  $(BitStr{N}(nzi - 1)) = $nzv")
        i != NNZ && println(io)
    end
end

bst_zeros(::Type{Tv}, ::Type{Ti}, N::Int) where {Tv,Ti} = BinarySparseTensor{Tv,Ti,N}(Dict{Ti, Tv}())
bst_zeros(::Type{Tv}, N::Int) where {Tv} = bst_zeros(Tv, Int64, N)

Base.zero(t::BinarySparseTensor{Tv,Ti,N}) where {Tv,Ti,N} = bst_zeros(Tv, Ti, N)
Base.copy(t::BinarySparseTensor{Tv,Ti,N}) where {Tv,Ti,N} = BinarySparseTensor{Tv,Ti,N}(copy(t.data))

function bstrand(ndim::Int, density::Real)
    BinarySparseTensor(sprand(1<<ndim, density))
end

function Base.permutedims!(dest::BinarySparseTensor{Tv,Ti,N}, src::BinarySparseTensor{Tv,Ti,N}, dims::NTuple{N,Int}) where {Tv,Ti,N}
    for (k, v) in src.data
        dest.data[bpermute(k-1, dims) + 1] = v
    end
    return dest
end
# permute bits in an integer
function bpermute(b::T, order) where T<:Integer
    res = zero(b)
    for (i, bi) in enumerate(order)
        res |= (b & bmask(T,bi)) >> (bi-i)
    end
    return res
end

function Base.permutedims(src::BinarySparseTensor{Tv,Ti,N}, dims) where {Tv,Ti,N}
    @assert length(dims) == N "dims should have length $N, got $(length(dims))"
    issorted(dims) && return src
    dest = zero(src)
    return Base.permutedims!(dest, src, (dims...,))
end