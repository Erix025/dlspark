from dlspark.auto_grad import Tensor, TensorOp, NDArray
class Transform:
    def __call__(self, x):
        raise NotImplementedError
    
class ToTensor(Transform):
    def __call__(self, x):
        if isinstance(x, NDArray):
            return Tensor(x)
        if isinstance(x, Tensor):
            return x
        raise TypeError("Unsupported type")
    