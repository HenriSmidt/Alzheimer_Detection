import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset

class MRIDataset(TorchDataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        images = [Image.open(path) for path in sample['paths']]
        class_label = sample['class']
        
        if self.transform:
            images = [self.transform(img) for img in images]
        
        return images, class_label

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, json_path, batch_size=4, transform=None):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.transform = transform
        self.data = pd.read_json(json_path)

    def setup(self, stage=None):
        # Splitting the data into training and testing
        train_subjects, test_subjects = train_test_split(self.data['subject_ID'].unique(), test_size=0.2, random_state=42)
        train_data = self.data[self.data['subject_ID'].isin(train_subjects)]
        test_data = self.data[self.data['subject_ID'].isin(test_subjects)]
        
        self.train_dataset = MRIDataset(train_data, self.transform)
        self.test_dataset = MRIDataset(test_data, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

