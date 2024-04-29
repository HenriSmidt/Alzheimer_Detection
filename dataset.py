import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset


class MRIDataset(pl.LightningDataModule):
    def __init__(self, json_path, batch_size=32, transform=None):
        super().__init__()
        self.json_path = json_path
        self.batch_size = batch_size
        self.transform = transform
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        # Load data and perform any necessary preprocessing
        pass

    def setup(self, stage=None):
        # Split data into training, validation, and testing sets
        data = pd.read_json(self.json_path)
        train_subjects, test_subjects = train_test_split(data['subject_ID'].unique(), test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(data[data['subject_ID'].isin(train_subjects)], test_size=0.2, random_state=42)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = data[data['subject_ID'].isin(test_subjects)]

    def train_dataloader(self):
        if self.train_data is None:
            self.setup()  # Ensure setup has been called
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        train_dataset = MRIPytorchDataset(self.train_data, transform=transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.val_data is None:
            self.setup()  # Ensure setup has been called
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        val_dataset = MRIPytorchDataset(self.val_data, transform=transform)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        if self.test_data is None:
            self.setup()  # Ensure setup has been called
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        test_dataset = MRIPytorchDataset(self.test_data, transform=transform)
        return DataLoader(test_dataset, batch_size=self.batch_size)

class MRIPytorchDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
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

# Example usage:
dataset = MRIDataset("Data/alzheimer_data.json", batch_size=8)

# Example: Accessing a sample
for images, class_label in dataset.train_dataloader():
    print("No of Slices:", images.size(0))
    print("Class Label:", class_label)
    break
