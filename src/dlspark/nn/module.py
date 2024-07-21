from dlspark.auto_grad import Tensor

class Parameter(Tensor):
    """ A kind of Tensor that is to be considered a module parameter.
    """

def _unpack_parameters(value):
    """ Return a list of parameters from a value."""
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, Module):
        return value.parameters()
    if isinstance(value, dict):
        params = []
        for v in value.values():
            params.extend(_unpack_parameters(v))
        return params
    if isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params.extend(_unpack_parameters(v))
        return params
    return []

def _unpack_modules(value):
    """ Return a list of modules from a value."""
    if isinstance(value, Module):
        return [value]
    if isinstance(value, dict):
        modules = []
        for v in value.values():
            modules.extend(_unpack_modules(v))
        return modules
    if isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules.extend(_unpack_modules(v))
        return modules
    return []

class Module:
    """ Base class for all neural network modules.
    """
    training : bool # Whether the module is in training mode.
    def __init__(self):
        self.training = True
    
    def parameters(self):
        """ Returns all module parameters. Only attributes that are instances of Parameter are considered parameters.
        """
        return _unpack_parameters(self.__dict__)
    
    def _children(self):
        """ Returns an iterator over immediate children modules."""
        return _unpack_modules(self.__dict__)
    
    def train(self):
        """ Set the module in training mode."""
        self.training = True
        for module in self._children():
            module.train()
            
    def eval(self):
        """ Set the module in evaluation mode."""
        self.training = False
        for module in self._children():
            module.eval()
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """ Call the module on input."""
        return self.forward(*args, **kwargs)
    
    