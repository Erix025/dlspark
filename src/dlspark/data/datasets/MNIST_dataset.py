from .dataset_basic import Dataset
import gzip
import numpy as np
import struct
class MNISTDataset(Dataset):
    images: np.ndarray
    labels: np.ndarray
    
    
    
    def __init__(
        self,
        image_file: str = None,
        label_file: str = None,
        train: bool = True,
        dataset_path: str = None,
        transforms=None
    ):
        super().__init__(transforms)
        if (image_file is None or label_file is None) and dataset_path is None:
            raise ValueError("Either provide image_file and label_file or dataset_path")
        
        if dataset_path is not None:
            if train:
                image_file = f"{dataset_path}/train-images-idx3-ubyte.gz"
                label_file = f"{dataset_path}/train-labels-idx1-ubyte.gz"
            else:
                image_file = f"{dataset_path}/t10k-images-idx3-ubyte.gz"
                label_file = f"{dataset_path}/t10k-labels-idx1-ubyte.gz"
        
        with gzip.open(image_file, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                num_images, rows, cols
            )
            self.images = self.images.astype(np.float32) / 255.0
        with gzip.open(label_file, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
            
    def __getitem__(self, idx):
        image = self.apply_transform(self.images[idx]).reshape(-1,1, 28, 28)
        label = self.apply_transform(np.array(self.labels[idx]))
        return image, label
    
    def __len__(self):
        return self.images.shape[0]