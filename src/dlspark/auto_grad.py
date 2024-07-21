"""Core data structures."""

import dlspark
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy
import networkx as nx
import matplotlib.pyplot as plt

NDArray = numpy.ndarray


class TensorOp:
    """Operator definition."""

    def compute(self, *args: Tuple[NDArray]):
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Tensor", node: "Tensor"
    ) -> Union["Tensor", Tuple["Tensor"]]:
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Tensor", node: "Tensor") -> Tuple["Tensor"]:
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

    def __call__(self, *args):
        # print("call:", type(args))
        return Tensor.make_from_op(self, args)

class Tensor:
    # trace of computational graph
    op: Optional[TensorOp]
    inputs: List["Tensor"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool
    grad: "Tensor"

    def __init__(
        self,
        array,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        cached_data = None
        if isinstance(array, Tensor):
            if dtype is None:
                dtype = array.dtype
            else:
                # fall back, copy through numpy conversion
                cached_data = numpy.array(array.data, dtype=dtype)
        else:
            cached_data = numpy.array(array, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    def _init(
        self,
        op: Optional[TensorOp],
        inputs: List["Tensor"],
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @staticmethod
    def make_from_op(op: TensorOp, inputs: List["Tensor"]):
        tensor = Tensor.__new__(Tensor)
        # print("make from op", type(inputs))
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None, 
            [], 
            cached_data=(
                data if not isinstance(data, Tensor) else data.realize_cached_data()
            ), 
            requires_grad=requires_grad)
        return tensor

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        if self.op is None:
            # print("Realize None", type(self.cached_data))
            return self.cached_data
        # avoid recomputation
        if self.cached_data is not None:
            # print("Realize Cached")
            return self.cached_data
        # note: data implicitly calls realized cached data
        # print("realize", self.inputs)
        # for x in self.inputs:
            # print("inputs", type(x))
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        # print("cache_data", id(self.cached_data))
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype
    
    @property
    def data(self):
        return self.realize_cached_data()
    
    @data.setter
    def data(self, value):
        self.cached_data = value

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else Tensor(numpy.ones_like(self.realize_cached_data(), dtype=self.dtype))
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "dlspark.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        
        return self.realize_cached_data()

    # Overload operators
    def __add__(self, other):
        if isinstance(other, Tensor):
            return dlspark.ops.EWiseAdd()(self, other)
        else:
            return dlspark.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return dlspark.ops.EWiseMul()(self, other)
        else:
            return dlspark.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return dlspark.ops.EWisePow()(self, other)
        else:
            return dlspark.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return dlspark.ops.EWiseAdd()(self, dlspark.ops.Negate()(other))
        else:
            return dlspark.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return dlspark.ops.EWiseDiv()(self, other)
        else:
            return dlspark.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return dlspark.ops.MatMul()(self, other)

    def matmul(self, other):
        return dlspark.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return dlspark.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return dlspark.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return dlspark.ops.Reshape(shape)(self)

    def __neg__(self):
        return dlspark.ops.Negate()(self)

    def transpose(self, axes=None):
        return dlspark.ops.Transpose(axes)(self)

    def dump_graph(self):
        dump_graph(self)
    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for tensor in reverse_topo_order:
        tensor.grad = _sum_node_list(node_to_output_grads_list[tensor])
        # Propagate gradient contributions to each input node.
        if tensor.is_leaf():
            continue
        input_grads = list(tensor.op.gradient_as_tuple(tensor.grad, tensor))
        for i in range(len(tensor.inputs)):
            input_tensor = tensor.inputs[i]
            if input_tensor not in node_to_output_grads_list:
                node_to_output_grads_list[input_tensor] = []
            node_to_output_grads_list[input_tensor].append(input_grads[i])

def dump_graph(output_tensor: Tensor):
    """Draw the computational graph of the output node."""
    G = nx.DiGraph()
    node_to_id = {}
    node_to_layer = {}
    id_to_node = {}
    node_id = 0
    layer_to_nodes = {}
    
    queue = [output_tensor]
    
    """ Init 0 node """
    node_to_id[output_tensor] = node_id
    id_to_node[node_id] = output_tensor
    G.add_node(node_to_id[output_tensor])
    node_to_layer[output_tensor] = 0
    layer_to_nodes[0] = [node_to_id[output_tensor]]
    node_id += 1
    
    while queue:
        node = queue.pop(0)
        layer = node_to_layer[node]
        for input_node in node.inputs:
            if input_node not in node_to_id:
                """ add node to graph """
                node_to_id[input_node] = node_id
                id_to_node[node_id] = input_node
                node_id += 1
                G.add_node(node_to_id[input_node])
            if input_node not in node_to_layer:
                """ add edge to layer """
                node_to_layer[input_node] = layer+1
                if layer+1 not in layer_to_nodes:
                    layer_to_nodes[layer+1] = []
                layer_to_nodes[layer+1].append(node_to_id[input_node])
                label = str(type(node.op))
                label = label[label.rfind('.')+1:label.rfind("'")]
                G.add_edge(node_to_id[input_node], node_to_id[node], label=label)
                queue.append(input_node)

    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')
    pos = nx.multipartite_layout(G, layer_to_nodes)
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=150, node_color="skyblue", font_size=15, font_color="black")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)
    plt.savefig(str(id(output_tensor)) + '.png', format='png')
    plt.clf()
    plt.close()

def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """

    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    inputs = node.inputs
    for input_node in inputs:
        if input_node not in visited:
            topo_sort_dfs(input_node, visited, topo_order)
    visited.add(node)
    topo_order.append(node)

def _sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
