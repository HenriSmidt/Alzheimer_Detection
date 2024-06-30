import unittest
import torch
import pandas as pd
import numpy as np
from utils import set_reproducibility
from dataset import MRIImageDataModule


class TestDataModuleSplit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.csv_path = "Data/metadata_for_preprocessed_files.csv"

    def test_train_val_test_split_consistency(self):
        # Set reproducibility
        set_reproducibility(42)

        # Initialize the DataModule
        data_module = MRIImageDataModule(
            data_path=self.csv_path, slice_number=65, batch_size=32
        )
        data_module.setup()

        # Get the datasets
        train_dataset_1 = data_module.train_dataset
        val_dataset_1 = data_module.val_dataset
        test_dataset_1 = data_module.test_dataset

        # Save the IDs from the first run
        train_ids_1 = set(train_dataset_1.valid_ids)
        val_ids_1 = set(val_dataset_1.valid_ids)
        test_ids_1 = set(test_dataset_1.valid_ids)

        # Reset reproducibility and reinitialize the DataModule to check consistency
        set_reproducibility(42)
        data_module = MRIImageDataModule(
            data_path=self.csv_path, slice_number=86, batch_size=32
        )
        data_module.setup()

        # Get the datasets again
        train_dataset_2 = data_module.train_dataset
        val_dataset_2 = data_module.val_dataset
        test_dataset_2 = data_module.test_dataset

        # Save the IDs from the second run
        train_ids_2 = set(train_dataset_2.valid_ids)
        val_ids_2 = set(val_dataset_2.valid_ids)
        test_ids_2 = set(test_dataset_2.valid_ids)

        # Check if the splits are the same
        self.assertEqual(train_ids_1, train_ids_2, "Train IDs do not match")
        self.assertEqual(val_ids_1, val_ids_2, "Validation IDs do not match")
        self.assertEqual(test_ids_1, test_ids_2, "Test IDs do not match")


if __name__ == "__main__":
    unittest.main()
