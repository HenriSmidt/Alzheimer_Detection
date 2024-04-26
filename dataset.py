import os
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = self._get_image_paths()
        self.groups = self._group_images_by_session_mpr()

    def _get_image_paths(self):
        image_paths = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):
                    image_paths.append(os.path.join(class_dir, filename))
        return image_paths

    def _group_images_by_session_mpr(self):
        groups = defaultdict(list)
        for img_path in self.image_paths:
            basename = os.path.basename(img_path)
            session_mpr = '_'.join(basename.split('_')[2:4])
            groups[session_mpr].append(img_path)
        return groups

    def _split_data_by_patients(self, test_size=0.2, shuffle=True, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
        patients = set([os.path.basename(img_path).split('_')[1] for img_path in self.image_paths])
        patients = list(patients)
        np.random.shuffle(patients)
        split_idx = int(len(patients) * (1 - test_size))
        train_patients = patients[:split_idx]
        test_patients = patients[split_idx:]
        return train_patients, test_patients

    def _get_indices_by_patients(self, patients):
        indices = []
        for patient_id in patients:
            patient_indices = [idx for idx, img_path in enumerate(self.image_paths) if patient_id in os.path.basename(img_path)]
            indices.extend(patient_indices)
        return indices

    def _get_subset_sampler(self, indices):
        return SubsetRandomSampler(indices)

    def __len__(self):
        return sum(len(group) for group in self.groups.values())

    def __getitem__(self, idx):
        group_paths = list(self.groups.values())[idx]
        images = [Image.open(img_path) for img_path in group_paths]
        
        if self.transform:
            images = [self.transform(img) for img in images]

        # Extract class label from the folder name
        label = os.path.basename(os.path.dirname(group_paths[0]))

        return images, label

    def get_data_loaders(self, test_size=0.2, batch_size=4, shuffle=True, random_seed=None):
        train_patients, test_patients = self._split_data_by_patients(test_size, shuffle, random_seed)
        train_indices = self._get_indices_by_patients(train_patients)
        test_indices = self._get_indices_by_patients(test_patients)

        train_sampler = self._get_subset_sampler(train_indices)
        test_sampler = self._get_subset_sampler(test_indices)

        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler, collate_fn=self._collate_fn)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler, collate_fn=self._collate_fn)

        return train_loader, test_loader

    def _collate_fn(self, batch):
        images, labels = zip(*batch)
        images = [torch.stack(imgs) for imgs in images]
        return images, labels


# Example usage:
if __name__ == "__main__":
    # Define data directory and transformation
    data_dir = "Data"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset instance
    dataset = MRIDataset(root_dir=data_dir, transform=transform)

    # Create data loaders
    train_loader, test_loader = dataset.get_data_loaders(test_size=0.2, batch_size=4, shuffle=True, random_seed=42)

    # Iterate through the training dataset
    for images, labels in train_loader:
        print("Training batch:", len(images), labels)

    # Iterate through the test dataset
    for images, labels in test_loader:
        print("Test batch:", len(images), labels)
