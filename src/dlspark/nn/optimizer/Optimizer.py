from dlspark.auto_grad import Tensor

class Optimizer:
    def __init__(self, params: list["Tensor"]):
        """
        params : 1D list
            A list of parameters(Tensors) to optimize.
        """
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = None