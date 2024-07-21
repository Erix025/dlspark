from dlspark.nn.optimizer import Optimizer
from dlspark.auto_grad import Tensor
import numpy as np

class Adam(Optimizer):
    def __init__(self,
                 params: list["Tensor"],
                 lr: float = 0.001,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 amsgrad: bool = False):
        super().__init__(params)
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.step_t = 0 # For unbias

        self.u = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.v = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.v_max = [Tensor(np.zeros_like(p.data)) for p in self.params]
        
    def step(self):
        self.step_t += 1

        bias_correction1 = 1 - self.beta1 ** self.step_t
        bias_correction2 = 1 - self.beta2 ** self.step_t

        for i, (p, u, v) in enumerate(zip(self.params, self.u, self.v)):
            u.data = self.beta1 * u.data + (1 - self.beta1) * p.grad.data
            v.data = self.beta2 * v.data + (1 - self.beta2) * np.square(p.grad.data)

            u_hat = u.data / bias_correction1
            v_hat = v.data / bias_correction2

            if self.amsgrad:
                self.v_max[i].data = np.maximum(self.v_max[i].data, v_hat)
                denom = np.sqrt(self.v_max[i].data) + self.eps
            else:
                denom = np.sqrt(v_hat) + self.eps
            
            p.data -= Tensor(self.lr * u_hat / denom).data
