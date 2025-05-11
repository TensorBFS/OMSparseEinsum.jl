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

# 4.66 -> 0.27
@time net.code(dense...)

# 5.88 -> 4.18
@time net.code(tensors...)