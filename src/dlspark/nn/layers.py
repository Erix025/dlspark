from dlspark.nn.module import Module, Parameter
from dlspark.auto_grad import Tensor
from dlspark import ops
import dlspark.init as init

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X)
        return X

class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weight and bias
        self.weight = Parameter(
            init.normal((in_features, out_features),in_features).numpy()
        )
        self.bias = (
            Parameter(
                init.zeros((out_features,1))
                .transpose().numpy()
            )
            if bias
            else None
        )
        # print("Linear weight shape: ", self.weight.shape)
        # print("Linear bias shape: ", self.bias.shape)
        
    def forward(self, X: Tensor) -> Tensor:
        # Perform the linear transformation
        # print(type(self.weight))
        Ax = ops.matmul(X, self.weight)
        if self.bias is not None:
            Ax = Ax + ops.broadcast_to(self.bias, Ax.shape)
        return Ax


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize weight and bias
        self.weight = Parameter(
            init.normal((out_channels, in_channels, kernel_size, kernel_size), out_channels*in_channels*kernel_size*kernel_size).numpy()
        )
        self.bias = (
            Parameter(
                init.zeros((out_channels,1))
                .transpose().numpy()
            )
            if bias
            else None
        )
        

    def forward(self, X: Tensor) -> Tensor:
        # if self.bias is not None:
        #     db = numpy.einsum('ijkl->j', out_grad)
        # Perform the convolution
        out = ops.conv2d(X, self.weight, self.bias,stride=self.stride, padding=self.padding)
        # if self.bias is not None:
        #     out = out + ops.broadcast_to(self.bias, out.shape)
        return out
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, X: Tensor) -> Tensor:
        # Perform the max pooling
        # TODO: Implement MaxPool2d forward
        return ops.max_pool2d(X, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    
class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        # Perform the flattening
        return ops.reshape(X, (X.shape[0], -1))
    
class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, X: Tensor) -> Tensor:
        # Perform the average pooling
        return ops.avg_pool2d(X, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)