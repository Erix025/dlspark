from dlspark.nn.module import Module
from dlspark.auto_grad import Tensor
from dlspark import ops

class ReLU(Module):
    def forward(self, X: Tensor) -> Tensor:
        return ops.relu(X)