using GenericTensorNetworks
using SparseTN

function mis3_network(n::Int)
    graph = random_regular_graph(n, 3)
    problem = IndependentSet(graph)
    net = GenericTensorNetwork(problem)
    return net
end

net = mis3_network(10)
dense = GenericTensorNetworks.generate_tensors(1.0, net)
tensors = map(t -> BinarySparseTensor(t), dense)
net.code(dense...)
net.code(tensors...)