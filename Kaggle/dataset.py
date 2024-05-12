import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning.pytorch as pl
from sklearn.model_selection import train_test_split
from torchvision.io import read_image

class MRIDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a common size
            transforms.ToTensor()           # Convert image to tensor
        ])
        self.class_to_index = {'Non Demented': 0, 'Very mild Dementia': 1, 'Mild Dementia': 2, 'Moderate Dementia': 3}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        class_label = self.class_to_index[sample['class']]
        
        images = []
        for path in sample['paths']:
            with Image.open(path) as img:  # Ensure files are closed after opening
                img = img.convert('RGB')  # Convert to RGB if not already in RGB
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        images_tensor = torch.stack(images, dim=0)  # Stack images along a new dimension
        return images_tensor, class_label


class MRIDataModule(pl.LightningDataModule):
    def __init__(self, json_path, batch_size=4, transform=None):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.transform = transform
        self.data = pd.read_json(json_path)

    def setup(self, stage=None):
        # Splitting the data to ensure no subject is in both train and test sets
        train_subjects, test_subjects = train_test_split(self.data['subject_ID'].unique(), test_size=0.2, random_state=42)
        train_data = self.data[self.data['subject_ID'].isin(train_subjects)]
        test_data = self.data[self.data['subject_ID'].isin(test_subjects)]
        
        self.train_dataset = MRIDataset(train_data, self.transform)
        self.test_dataset = MRIDataset(test_data, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)