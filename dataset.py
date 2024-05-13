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
    

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import random

class MRIDataset(Dataset):
    def __init__(self, dataframe, slice_number, transform=None):
        self.df = dataframe.set_index(['ID', 'slice_number'])
        self.slice_number = slice_number
        self.transform = transform

    def __len__(self):
        return len(self.df.index.unique())

    def __getitem__(self, idx):
        id, slice_num = self.df.index.unique()[idx]
        images = []
        for offset in (-1, 0, 1):
            slice_path = self.get_random_path(id, slice_num + offset)
            image = Image.open(slice_path).convert('L')
            if self.transform:
                image = self.transform(image)
            images.append(np.array(image))
        
        # Stack to create a 3-channel image
        image_stack = np.stack(images, axis=-1)
        return image_stack

    def get_random_path(self, id, slice_num):
        try:
            rows = self.df.loc[(id, slice_num)]
            if not rows.empty:
                # Randomly select between masked and unmasked if available
                row = rows.sample(n=1)
                return row['path'].values[0]
            else:
                # If the specific slice_num doesn't exist, default to the original slice
                return self.df.loc[(id, self.slice_number)].sample(n=1)['path'].values[0]
        except KeyError:
            # In case the slice is completely unavailable, use the fallback slice
            return self.df.loc[(id, self.slice_number)].sample(n=1)['path'].values[0]

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

