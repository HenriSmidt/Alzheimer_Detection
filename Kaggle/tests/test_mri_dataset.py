import unittest
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from dataset import (
    MRIDataset,
    MRIDataModule,
)  # Ensure your classes are correctly imported


class TestMRIDataWithRealData(unittest.TestCase):
    def setUp(self):
        # Path to your actual JSON file
        self.json_path = "Data/alzheimer_data.json"
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize to a common size
                transforms.ToTensor(),  # Convert image to tensor
            ]
        )
        self.data_module = MRIDataModule(self.json_path, transform=self.transform)
        self.data_module.setup()  # This sets up train and test datasets

    def test_subject_leakage(self):
        # Check that no subject appears in both the training and testing sets
        train_subjects = set(self.data_module.train_dataset.data["subject_ID"].unique())
        test_subjects = set(self.data_module.test_dataset.data["subject_ID"].unique())
        self.assertTrue(
            train_subjects.isdisjoint(test_subjects),
            "Subjects should not overlap between train and test sets",
        )

    def test_complete_session_load(self):
        # Check if all slices for each session and mpr are loaded
        for dataset in [self.data_module.train_dataset, self.data_module.test_dataset]:
            for index in range(len(dataset.data)):
                sample = dataset.data.iloc[index]
                expected_slices = list(range(100, 161))  # Expected slice numbers
                actual_slices = [
                    int(path.split("_")[-1].split(".")[0]) for path in sample["paths"]
                ]
                self.assertEqual(
                    expected_slices,
                    actual_slices,
                    "Not all slices from a session are loaded",
                )


if __name__ == "__main__":
    unittest.main()
