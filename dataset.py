import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import numpy as np
from utils import get_best_device
import pickle
from sampler import create_generic_weighted_sampler


# Function to perform stratified split based on groups
def stratified_group_split(
    data, group_col, stratify_col, test_size=0.125, random_state=42
):
    unique_ids = data[group_col].unique()
    stratify_values = data.groupby(group_col)[stratify_col].first().values
    train_val_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_size,
        stratify=stratify_values,
        random_state=random_state,
    )
    return train_val_ids, test_ids


class MRIDataset(Dataset):
    def __init__(self, dataframe, slice_number, transform=None, return_id=False, soft_labels=None):
        self.slice_number = slice_number
        self.transform = transform
        self.valid_ids = dataframe[dataframe["slice_number"] == slice_number]["ID"].unique()
        self.df = dataframe[dataframe["ID"].isin(self.valid_ids)].set_index(["ID", "slice_number"]).sort_index()
        self.device = get_best_device()
        self.return_id = return_id
        self.label_map = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        id = self.valid_ids[idx]
        images = []
        for offset in (-1, 0, 1):
            slice_path = self.get_random_path(id, self.slice_number + offset)
            image = Image.open(slice_path).convert("L")
            image = np.array(image)
            images.append(image)

        image_stack = np.stack(images, axis=-1)
        image_stack = torch.tensor(image_stack).permute(2, 0, 1)
        image_stack = image_stack.to(self.device)

        if self.transform:
            image_stack = self.transform(image_stack)

        float_label = self.df.loc[(id, self.slice_number)]["CDR"].iloc[0]
        label = torch.tensor(self.label_map[float(float_label)]).long()
        age = torch.tensor(self.df.loc[(id, self.slice_number)]["Age"]).float()

        item = {
            'inputs': image_stack,
            'labels': label,
            'age': age
        }

        if self.soft_labels is not None:
            item['soft_labels'] = torch.tensor(self.soft_labels[id]).float()

        if self.return_id:
            item['id'] = id

        return item

    def get_random_path(self, id, slice_num):
        try:
            rows = self.df.loc[(id, slice_num)]
            if not rows.empty:
                # Randomly select between masked and unmasked if available
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row["path"]
                return (
                    sampled_path
                    if isinstance(sampled_path, str)
                    else sampled_path.values[0]
                )
            else:
                # If the specific slice_num doesn't exist, default to the original slice
                rows = self.df.loc[(id, self.slice_number)]
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row["path"]
                return (
                    sampled_path
                    if isinstance(sampled_path, str)
                    else sampled_path.values[0]
                )
        except KeyError:
            # In case the slice is completely unavailable, use the fallback slice
            try:
                rows = self.df.loc[(id, self.slice_number)]
                if isinstance(rows, pd.DataFrame) and len(rows) > 1:
                    row = rows.iloc[np.random.choice(len(rows))]
                else:
                    row = rows

                sampled_path = row["path"]
                return (
                    sampled_path
                    if isinstance(sampled_path, str)
                    else sampled_path.values[0]
                )
            except KeyError:
                print(
                    f"KeyError: The slice number {self.slice_number} or id {id} does not exist in the Data."
                )
        return None
    
    def get_labels(self):
        """
        Returns a list of labels for each sample in the dataset.
        This method is used by the weighted sampler to obtain the class distribution.
        """
        labels = [self.df.loc[(id, self.slice_number), "CDR"].iloc[0] for id in self.valid_ids]
        converted_labels = [self.label_map[float(label)] for label in labels]
        return converted_labels


class MRIImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        transform=None,
        batch_size=32,
        slice_number=87,
        num_workers=0,
        always_return_id=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.slice_number = slice_number
        self.transform = transform
        self.num_workers = num_workers
        self.always_return_id = always_return_id

    def setup(self, stage=None):
        data = pd.read_csv(self.data_path)

        # Filter to only include IDs with the specified slice_number
        data = data[
            data["slice_number"].isin(
                [self.slice_number - 1, self.slice_number, self.slice_number + 1]
            )
        ]

        # Initial stratified split: 87.5% train + validation, 12.5% test
        train_val_ids, test_ids = stratified_group_split(
            data, "ID", "CDR", test_size=0.125
        )
        

        # Creating the train + validation DataFrame for further splitting
        train_val_df = data[data["ID"].isin(train_val_ids)]

        # Further stratified split train + validation into 75% train and 12.5% validation (relative to the total dataset)
        unique_train_val_ids = train_val_df["ID"].unique()
        stratify_train_val_values = train_val_df.groupby("ID")["CDR"].first().values

        train_ids, val_ids = train_test_split(
            unique_train_val_ids,
            test_size=0.142857,
            stratify=stratify_train_val_values,
            random_state=42,
        )
        
        train_df = train_val_df[train_val_df["ID"].isin(train_ids)]
        val_df = train_val_df[train_val_df["ID"].isin(val_ids)]
        test_df = data[data["ID"].isin(test_ids)]

        # Assuming MRIDataset is a class that takes a DataFrame, slice number, and optional transform as arguments
        self.train_dataset = MRIDataset(
            train_df, self.slice_number, transform=self.transform, return_id=self.always_return_id
        )
        self.val_dataset = MRIDataset(
            val_df, self.slice_number, transform=self.transform, return_id=self.always_return_id
        )
        self.test_dataset = MRIDataset(
            test_df, self.slice_number, transform=self.transform, return_id=True
        )

    def train_dataloader(self, sampling_strategy='sqrt', smoothing=0.0):
        """    Parameters:
                - strategy: The strategy for weight computation ('inverse', 'sqrt', 'log', 'exp', 'custom') (default: 'sqrt')
                - smoothing: A smoothing factor to add to class counts of the sampler to avoid harsh weights (default: 0.0)
        """
        sampler = create_generic_weighted_sampler(self.train_dataset, strategy=sampling_strategy, smoothing=smoothing)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# Usage example
# data_module = MRIImageDataModule(data_path='Data/metadata_for_preprocessed_files.csv', slice_number=63)
# data_module.setup()
# train_loader = data_module.train_dataloader()
# for batch in train_loader:
#     print(batch.shape)  # Should output torch.Size([batch_size, 3, 224, 224])

class MRIFeatureDataset(Dataset):
    def __init__(self, pickle_file, as_sequence):
        with open(pickle_file, 'rb') as f:
            self.data =  pickle.load(f)
        
        self.as_sequence = as_sequence
        # Identify feature columns and sort them
        self.feature_columns = sorted([col for col in self.data.columns if col.startswith('slice_')])
        
        # Determine sequence length based on the number of feature columns
        self.sequence_length = len(self.feature_columns)
        
        # Determine the length of the individual feature maps
        self.featuremap_length = len(self.data[self.feature_columns[0]][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.as_sequence:
            # Initialize sequence with zeros for missing slices
            sequence = np.zeros((self.sequence_length, self.featuremap_length))
            
            for i, col in enumerate(self.feature_columns):
                feature_map = row[col]
                sequence[i] = feature_map

            sequence = torch.tensor(sequence, dtype=torch.float32)
            return sequence, label
        else: 
            arrays = [row[col] for col in self.feature_columns]
            features = np.concatenate(arrays, axis=0)
            features = torch.tensor(features, dtype=torch.float32)
            return features, label
        
    def get_labels(self):
        """
        Returns a list of labels for each sample in the dataset.
        This method is used by the weighted sampler to obtain the class distribution.
        """
        return self.data['label'].tolist()    
    
            

class MRIFeatureDataModule(pl.LightningDataModule):
    def __init__(self, train_pkl, val_pkl, test_pkl, as_sequence=False, batch_size=32, num_workers=0):
        super().__init__()
        self.train_pkl = train_pkl
        self.val_pkl = val_pkl
        self.test_pkl = test_pkl
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.as_sequence = as_sequence
        with open(val_pkl, 'rb') as f:
            data =  pickle.load(f)
            feature_columns = [col for col in data.columns if col.startswith('slice_')]
            self.sequence_length = len(feature_columns)
            self.featuremap_length = len(data[feature_columns[0]][0])

    def setup(self, stage=None):
        self.train_dataset = MRIFeatureDataset(self.train_pkl, as_sequence=self.as_sequence)
        self.val_dataset = MRIFeatureDataset(self.val_pkl, as_sequence=self.as_sequence)
        self.test_dataset = MRIFeatureDataset(self.test_pkl, as_sequence=self.as_sequence)

    def train_dataloader(self, sampling_strategy='sqrt', smoothing=0.0):
        """    Parameters:
                - strategy: The strategy for weight computation ('inverse', 'sqrt', 'log', 'exp', 'custom') (default: 'sqrt')
                - smoothing: A smoothing factor to add to class counts of the sampler to avoid harsh weights (default: 0.0)
        """
        sampler = create_generic_weighted_sampler(self.train_dataset, strategy=sampling_strategy, smoothing=smoothing)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

