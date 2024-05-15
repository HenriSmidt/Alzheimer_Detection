import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np
from utils import get_best_device

class MRIDataset(Dataset):
    def __init__(self, dataframe, slice_number, transform=None):
        self.slice_number = slice_number
        self.transform = transform
        self.df = dataframe
        self.valid_ids = dataframe[dataframe['slice_number'] == slice_number]['ID'].unique()
        self.df = self.df[self.df['ID'].isin(self.valid_ids)].set_index(['ID', 'slice_number'])
        self.device = get_best_device()

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        id = self.valid_ids[idx]
        images = []
        
        for offset in (-1, 0, 1):
            slice_path = self.get_random_path(id, self.slice_number + offset)
            image = Image.open(slice_path).convert('L')
            image = np.array(image)
            images.append(image)
        
        # Stack to create a 3-channel image
        image_stack = np.stack(images, axis=-1)  # Shape will be (H, W, C)
        image_stack = torch.tensor(image_stack).permute(2, 0, 1)  # Convert to (C, H, W) tensor
        
        # Normalize the image stack to [0, 1]
        image_stack = image_stack.float() / 255.0 #TODO: check if images are already normalized and sized to 224 224
        image_stack = image_stack.to(torch.device("mps"))
        
        if self.transform:
            image_stack = self.transform(image_stack)
            
        label = torch.tensor(self.df.loc[(id, self.slice_number)]['CDR']).float()
        age = torch.tensor(self.df.loc[(id, self.slice_number)]['Age']).float()
        
        label = label.to(torch.device("mps"))

            
        return image_stack, label, age

    def get_random_path(self, id, slice_num):
        try:
            rows = self.df.loc[(id, slice_num)]
            if not rows.empty:
                # Randomly select between masked and unmasked if available
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row['path']
                return sampled_path if isinstance(sampled_path, str) else sampled_path.values[0]
            else:
                # If the specific slice_num doesn't exist, default to the original slice
                rows = self.df.loc[(id, self.slice_number)]
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row['path']
                return sampled_path if isinstance(sampled_path, str) else sampled_path.values[0]
        except KeyError:
            # In case the slice is completely unavailable, use the fallback slice
            try:
                rows = self.df.loc[(id, self.slice_number)]
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row['path']
                return sampled_path if isinstance(sampled_path, str) else sampled_path.values[0]
            except KeyError:
                print(f"KeyError: The slice number {self.slice_number} or id {id} does not exist in the Data.")
        return None


class MRIImageDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, slice_number=87):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.slice_number = slice_number

    def setup(self, stage=None):
        data = pd.read_csv(self.data_path)
        
        # Filter to only include IDs with the specified slice_number
        data = data[data['slice_number'].isin([self.slice_number - 1, self.slice_number, self.slice_number + 1])]

        train_ids, test_ids = train_test_split(data['ID'].unique(), test_size=0.2, random_state=42)

        train_df = data[data['ID'].isin(train_ids)]
        test_df = data[data['ID'].isin(test_ids)]

        self.train_dataset = MRIDataset(train_df, self.slice_number)
        self.test_dataset = MRIDataset(test_df, self.slice_number)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Usage example
# data_module = MRIImageDataModule(data_path='Data/metadata_for_preprocessed_files.csv', slice_number=63)
# data_module.setup()
# train_loader = data_module.train_dataloader()
# for batch in train_loader:
#     print(batch.shape)  # Should output torch.Size([batch_size, 3, 224, 224])
