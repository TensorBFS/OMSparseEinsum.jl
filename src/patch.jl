BitBasis.log2i(x::LongLongUInt{C}) where C = floor(Int, log2(Float64(BigInt(x))))
Base.BigInt(x::LongLongUInt{C}) where C = mapfoldl(x -> BigInt(x), (x, y) -> ((x << 64) | y), x.content)