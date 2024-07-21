from ..transforms import Transform
from typing import Optional

class Dataset:
    
    def __init__(self, transforms):
        if transforms is None:
            self.transforms = []
        elif not isinstance(transforms, list):
            self.transforms = [transforms]
        else:
            self.transforms = transforms
        
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def apply_transform(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x