from dlspark.nn.module import Module
from dlspark.auto_grad import Tensor
from dlspark import ops
from dlspark import init

class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        logits : Tensor (Batch, Category)
            The input logits.
        target : Tensor (Batch)
            The target distribution.

        output = -sum(target * log(softmax(logits)))
        """
        logits = logits - ops.max(logits, axis=-1)
        target = init.onehot(target, logits.shape[-1])

        sum_exp = ops.log(ops.summation(ops.exp(logits), axes=(-1,)))
        p = ops.exp(logits - (sum_exp
                      .reshape(logits.shape[:-1] + (1,))
                      .broadcast_to(logits.shape)))
        return -ops.summation(target * ops.log(p)) / logits.shape[0]