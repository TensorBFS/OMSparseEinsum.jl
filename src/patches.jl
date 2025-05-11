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