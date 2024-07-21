from dlspark.auto_grad import TensorOp
import numpy

class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(a, 0)

    def gradient(self, out_grad, node):
        return out_grad * (node.realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)