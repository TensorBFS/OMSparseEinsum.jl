# binary sparse tensor
struct BinarySparseTensor{Tv,Ti<:Integer,N} <: AbstractSparseArray{Tv, Ti, N}
   data:: SparseVector{Tv, Ti}  # indices must be sorted!
end
is_healthy(t::BinarySparseTensor) = issorted(t.data.nzind)

function bst(data::SparseVector{T,Ti}) where {T,Ti}
    N = log2i(length(data))
    length(data) != one(Ti) << N && throw(ArgumentError("data length should be 2^x, got $(length(data))"))
    BinarySparseTensor{T,Ti,N}(data)
end

function BinarySparseTensor(A::AbstractArray)
    bst(SparseVector(vec(A)))
end
function BinarySparseTensor{Tv, Ti}(A::AbstractArray) where {Tv, Ti}
    sv = SparseVector(vec(A))
    BinarySparseTensor(SparseVector{Tv, Ti}(Ti(length(A)), Ti.(sv.nzind), sv.nzval))
end

function Base.getindex(t::BinarySparseTensor{T,Ti,N}, index::BitStr{N}) where {T,Ti,N}
    idx = as_index(Ti, index)
    @boundscheck idx <= length(t.data) || throw(BoundsError(t, index))
    @inbounds return t.data[idx]
end
function Base.getindex(t::BinarySparseTensor{T,Ti,N}, index::Int...) where {T,Ti,N}
    idx =as_index(Ti, index)
    @boundscheck idx <= length(t.data) || throw(BoundsError(t, index))
    @inbounds return t.data[idx]
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

function Base.setindex!(t::BinarySparseTensor{T,Ti,N}, val, index::BitStr{N,Ti}) where {T,Ti,N}
    return t.data[as_index(Ti, index)] = val
end
function Base.setindex!(t::BinarySparseTensor{T,Ti,N}, val, index::Integer) where {T, Ti, N}
    @boundscheck one(Ti) <= index <= one(Ti)<<N || throw(BoundsError(t, index))
    return @inbounds t.data[as_index(Ti, index)] = val
end

SparseArrays.nnz(t::BinarySparseTensor) = nnz(t.data)
function SparseArrays.findnz(t::BinarySparseTensor{Tv,Ti,N}) where {Tv,Ti,N}
    return t.data.nzind, t.data.nzval
end
# SparseArrays.nonzeroinds(t::BinarySparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = convert.(BitStr{N,Ti}, t.data.nzind.-1)
# SparseArrays.nonzeros(t::BinarySparseTensor{Tv,Ti,N}) where {Tv, Ti, N} = t.data.nzval
Base.Array(t::BinarySparseTensor{Tv,Ti,1}) where {Tv,Ti} = Base.Array(t.data)

Base.show(io::IO, ::MIME"text/plain", t::BinarySparseTensor) = Base.show(io, t)
function Base.show(io::IOContext, t::BinarySparseTensor{T,Ti,1}) where {T,Ti}
    invoke(show, Tuple{IO,BinarySparseTensor}, io, t)
end
function Base.show(io::IO, t::BinarySparseTensor{T,Ti,N}) where {T,Ti,N}
    NNZ = length(t.data.nzind)
    println(io, "$(summary(t)) with $(nnz(t)) stored entries:")
    for (i, (nzi, nzv)) in enumerate(zip(findnz(t)...))
        print(io, "  $(BitStr{N}(nzi - 1)) = $nzv")
        i != NNZ && println(io)
    end
end

bst_zeros(::Type{Tv}, ::Type{Ti}, N::Int) where {Tv,Ti} = BinarySparseTensor{Tv,Ti,N}(SparseVector(1<<N, Ti[], Tv[]))
bst_zeros(::Type{Tv}, N::Int) where {Tv} = bst_zeros(Tv, Int64, N)

Base.zero(t::BinarySparseTensor) = bst(SparseVector(t.data.n, t.data.nzind, zero(t.data.nzval)))
Base.copy(t::BinarySparseTensor) = bst(SparseVector(t.data.n, copy(t.data.nzind), copy(t.data.nzval)))

function bstrand(ndim::Int, density::Real)
    bst(sprand(1<<ndim, density))
end

function Base.permutedims!(dest::BinarySparseTensor{Tv,Ti,N}, src::BinarySparseTensor{Tv,Ti,N}, dims::NTuple{N,Int}) where {Tv,Ti,N}
    dest.data.nzind .= bpermute.(src.data.nzind .- 1, Ref(dims)) .+ 1
    order = sortperm(dest.data.nzind)
    @inbounds dest.data.nzind .= dest.data.nzind[order]
    @inbounds dest.data.nzval .= getindex.(Ref(src.data.nzval), order)
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

# the following two functions are used in sortperm function.
# function Base.sub_with_overflow(x::T, y::T) where T<:BitStr
#     res, sig = Base.sub_with_overflow(buffer(x), buffer(y))
#     return T(res), sig
# end
# function Base.add_with_overflow(x::T, y::T) where T<:BitStr
#     res, sig = Base.add_with_overflow(buffer(x), buffer(y))
#     return T(res), sig
# end

function Base.permutedims(src::BinarySparseTensor{Tv,Ti,N}, dims) where {Tv,Ti,N}
    issorted(dims) && return src
    dest = copy(src)
    return Base.permutedims!(dest, src, (dims...,))
end