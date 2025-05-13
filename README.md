# OMSparseEinsum

[![Build Status](https://github.com/TensorBFS/OMSparseEinsum.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TensorBFS/OMSparseEinsum.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/TensorBFS/OMSparseEinsum.jl/graph/badge.svg?token=fFtv7OCNuG)](https://codecov.io/gh/TensorBFS/OMSparseEinsum.jl)

OMSparseEinsum is a Julia package for efficient sparse tensor network computations. It provides einsum notations for binary sparse tensors.

## Installation

You can install OMSparseEinsum using Julia's package manager:

```julia
] dev https://github.com/TensorBFS/OMSparseEinsum.jl.git
```

## Usage
```julia
using OMSparseEinsum, OMEinsum

# construct the tensor network
code = EinCode([[1, 2], [2, 3], [3, 4]], [1, 4])
tensors = [SparseTensor([1 1; 1 0]) for _ in 1:3]

# optimize the contraction order
size_dict = OMEinsum.get_size_dict(code.ixs, tensors)
optcode = optimize_code(code, size_dict, TreeSA())

# compute the result
res = optcode(tensors...)

# output:
# 2×2 SparseTensor{Int64, Int64, 2} with 4 stored entries:
#   (2, 2) => 1
#   (2, 1) => 2
#   (1, 2) => 2
#   (1, 1) => 3
```

Please refer to the [examples](example/) for more usage examples.
