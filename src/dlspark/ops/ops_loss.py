from ..auto_grad import Tensor, TensorOp, NDArray
import numpy
from typing import Optional

class Softmax(TensorOp):
    def compute(self, a: NDArray):
        exp_a = numpy.exp(a - a.max(axis=1, keepdims=True))
        return exp_a / exp_a.sum()

    def gradient(self, out_grad: Tensor, node: Tensor):
        input = node.inputs[0]
        return out_grad * (Tensor(self.compute(input.realize_cached_data())) - out_grad)

def softmax(a) -> Tensor:
    return Softmax()(a)