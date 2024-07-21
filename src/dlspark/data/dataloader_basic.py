from .datasets import *
from dlspark.auto_grad import Tensor

class DataLoader:
    
    dataset: Dataset
    batch_size: int
    
    def __init__(self, dataset: Dataset, batch_size: int=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter_len = len(self.dataset) // self.batch_size
        
    def __iter__(self):
        self.index = 0
        self.iter_len = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        self.ordering = np.arange(len(self.dataset))
        return self
    
    def __len__(self):
        return self.iter_len
    
    def __next__(self):
        if self.index >= self.iter_len:
            raise StopIteration
        start_index = self.index * self.batch_size
        end_index = min((self.index + 1) * self.batch_size, len(self.dataset))
        self.index += 1
        batch = self.dataset[self.ordering[start_index:end_index]]
        batch = tuple(Tensor.make_const(item) for item in batch)
        return batch