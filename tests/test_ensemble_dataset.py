import unittest
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import tempfile
import os
from dataset import MRIFeatureDataset, MRIFeatureDataModule


class TestMRIFeatureDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()

        # Sample data
        self.data = {
            "slice_0": [np.random.rand(5) for _ in range(10)],
            "slice_1": [np.random.rand(5) for _ in range(10)],
            "slice_2": [np.random.rand(5) for _ in range(10)],
            "label": np.random.randint(0, 2, size=10),
        }
        self.df = pd.DataFrame(self.data)

        # Save the data to a pickle file
        self.pickle_file = os.path.join(self.temp_dir.name, "sample_data.pkl")
        with open(self.pickle_file, "wb") as f:
            pickle.dump(self.df, f)

    def tearDown(self):
        # Cleanup the temporary directory
        self.temp_dir.cleanup()

    def test_data_loading(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=True)
        self.assertEqual(len(dataset), len(self.df))

    def test_feature_columns_identification(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=True)
        expected_columns = sorted(
            [col for col in self.df.columns if col.startswith("slice_")]
        )
        self.assertEqual(dataset.feature_columns, expected_columns)

    def test_sequence_and_feature_length_calculation(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=True)
        self.assertEqual(dataset.sequence_length, 3)
        self.assertEqual(dataset.featuremap_length, 5)

    def test_len_method(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=True)
        self.assertEqual(len(dataset), 10)

    def test_getitem_as_sequence(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=True)
        sequence, label = dataset[0]
        self.assertEqual(sequence.shape, (3, 5))
        self.assertEqual(label.shape, torch.Size([]))
        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

    def test_getitem_not_as_sequence(self):
        dataset = MRIFeatureDataset(self.pickle_file, as_sequence=False)
        features, label = dataset[0]
        self.assertEqual(features.shape, (15,))
        self.assertEqual(label.shape, torch.Size([]))
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

    def test_batch_loading_as_sequence(self):
        batch_size = 4
        data_module = MRIFeatureDataModule(
            train_pkl=self.pickle_file,
            val_pkl=self.pickle_file,
            test_pkl=self.pickle_file,
            as_sequence=True,
            batch_size=batch_size,
            num_workers=0,
        )
        data_module.setup()

        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        sequences, labels = batch

        self.assertEqual(sequences.shape, (batch_size, 3, 5))
        self.assertEqual(labels.shape, (batch_size,))
        self.assertIsInstance(sequences, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

    def test_batch_loading_not_as_sequence(self):
        batch_size = 4
        data_module = MRIFeatureDataModule(
            train_pkl=self.pickle_file,
            val_pkl=self.pickle_file,
            test_pkl=self.pickle_file,
            as_sequence=False,
            batch_size=batch_size,
            num_workers=0,
        )
        data_module.setup()

        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        features, labels = batch

        self.assertEqual(features.shape, (batch_size, 15))
        self.assertEqual(labels.shape, (batch_size,))
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
