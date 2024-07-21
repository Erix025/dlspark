from dlspark.auto_grad import Tensor
import numpy as np

def zeros(shape):
    """ Return a tensor of zeros with the given shape.
    """
    return Tensor.make_const(np.zeros(shape))

def ones(shape):
    """ Return a tensor of ones with the given shape.
    """
    return Tensor.make_const(np.ones(shape))

def randn(shape):
    """ Return a tensor of random values from a normal distribution with mean 0 and variance 1.
    """
    return Tensor.make_const(np.random.randn(*shape))

def onehot(indices, depth):
    """ Return a tensor of one-hot vectors.
    """
    if isinstance(indices, Tensor):
        indices = indices.numpy()
    return Tensor.make_const(np.eye(depth)[indices])

def normal(shape, coe=2):
    """ Return a tensor of random values from a normal distribution with the given mean and standard deviation.
    """
    return Tensor.make_const(np.random.randn(*shape) * np.sqrt(2 / coe))