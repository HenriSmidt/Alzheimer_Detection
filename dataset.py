import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

    
class PreprocessDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = read_image(path)  # Load image as a tensor
        image = transforms.Grayscale()(image) / 255.0  # Convert to grayscale and normalize
        return image
    

