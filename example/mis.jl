using GenericTensorNetworks
using SparseTN

function mis3_network(n::Int)
    graph = random_regular_graph(n, 3)
    problem = IndependentSet(graph)
    net = GenericTensorNetwork(problem)
    return net
end

using Random; Random.seed!(42)
net = mis3_network(106)
dense = GenericTensorNetworks.generate_tensors(1.0, net)
tensors = map(t -> BinarySparseTensor(t), dense)

# 0.27
@time net.code(dense...)

# 1.6s
@time net.code(tensors...)

ltensors = map(t -> BinarySparseTensor{Float64, LongLongUInt{2}}(t), dense)
# 2.72
@time net.code(ltensors...)

using Profile
Profile.clear()
@profile net.code(tensors...)

Profile.print(mincount=100, format=:flat)