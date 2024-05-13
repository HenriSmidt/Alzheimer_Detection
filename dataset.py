import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from PIL import Image
import numpy as np
import pytorch_lightning as pl

    
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
    


class MRIDataset(Dataset):
    def __init__(self, dataframe, slice_number, transform=None):
        self.df = dataframe
        self.slice_number = slice_number
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for offset in (-1, 0, 1):
            slice_path = self.get_path_for_slice(row['ID'], self.slice_number + offset, row['path'])
            image = Image.open(slice_path).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(np.array(image))
        
        # Stack to create a 3-channel image
        image_stack = np.stack(images, axis=-1)
        return image_stack

    def get_path_for_slice(self, id, slice_num, original_path):
        try:
            row = self.df[(self.df['ID'] == id) & (self.df['slice_number'] == slice_num)]
            if not row.empty:
                return row.iloc[0]['path']
            else:
                return original_path  # Fallback to the original slice if adjacent not found
        except:
            return original_path  # Fallback in case of any error

class MRIImageDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, slice_number=63):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.slice_number = slice_number
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def setup(self, stage=None):
        data = pd.read_csv(self.data_path)
        train_ids, test_ids = train_test_split(data['ID'].unique(), test_size=0.2, random_state=42)

        train_df = data[data['ID'].isin(train_ids)]
        test_df = data[data['ID'].isin(test_ids)]

        self.train_dataset = MRIDataset(train_df, self.slice_number, transform=self.transform)
        self.test_dataset = MRIDataset(test_df, self.slice_number, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Usage
data_module = MRIImageDataModule(data_path='Data/metadata_for_preprocessed_files.csv', slice_number=63)
data_module.setup()

