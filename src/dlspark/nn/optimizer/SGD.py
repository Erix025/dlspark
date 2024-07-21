from dlspark.nn.optimizer import Optimizer
from dlspark.auto_grad import Tensor
import numpy as np

class SGD(Optimizer):
    def __init__(self,
                 params: list["Tensor"],
                 lr: float = 0.001,
                 momentum: float = 0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.step_t = 0
        
    def step(self):
        self.step_t += 1
        bias_correction = 1 - self.momentum ** self.step_t

        for p, u in zip(self.params, self.u):
            if self.momentum != 0:
                u.data = self.momentum * u.data + (1 - self.momentum) * p.grad.data
                u_hat = u.data / bias_correction
                p.data -= Tensor(self.lr * u_hat).data
            else:
                p.data -= Tensor(self.lr * p.grad.data).data