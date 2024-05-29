import unittest
import pandas as pd
from torchvision import transforms
import sys
import os
import torch
import numpy as np
from dataset import stratified_group_split
from sklearn.model_selection import train_test_split

# Add the parent directory to sys.path to import dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import MRIImageDataModule, MRIDataset

# Your dataset and data module classes here (MRIDataset, MRIImageDataModule)
# Make sure to import or define them if they are in another file.


class TestMRIDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV for testing
        self.temp_csv_path = "temp_test_data.csv"
        data = {
            "ID": [
                "OAS1_0001_MR1",
                "OAS1_0001_MR1",
                "OAS1_0001_MR1",
                "OAS1_0002_MR1",
                "OAS1_0002_MR1",
                "OAS1_0003_MR1",
                "OAS1_0003_MR1",
                "OAS1_0001_MR1",
                "OAS1_0001_MR1",
                "OAS1_0001_MR1",
                "OAS1_0002_MR1",
                "OAS1_0002_MR1",
                "OAS1_0003_MR1",
                "OAS1_0003_MR1",
            ],
            "slice_number": [62, 63, 64, 63, 64, 63, 62, 62, 63, 64, 63, 64, 63, 62],
            "path": [
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_62.jpeg",
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg",
                "Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg",
                "Data/OASIS_Extracted/OAS1_0003/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0003/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_62.jpeg",
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg",
                "Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg",
                "Data/OASIS_Extracted/OAS1_0003/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
                "Data/OASIS_Extracted/OAS1_0003/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg",
            ],
            "CDR": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "is_masked": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "Age": [44, 55, 66, 77, 88, 99, 11, 44, 55, 66, 77, 88, 99, 11],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.temp_csv_path, index=False)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def tearDown(self):
        # Remove temporary CSV file
        os.remove(self.temp_csv_path)

    def test_dataset_length(self):
        module = MRIImageDataModule(self.temp_csv_path, batch_size=1, slice_number=63)
        module.setup()
        self.assertEqual(
            len(module.train_dataset)
            + len(module.test_dataset)
            + len(module.val_dataset),
            3,
        )  # Should have unique ID entries for training

    def test_data_loading(self):
        module = MRIImageDataModule(self.temp_csv_path, batch_size=1, slice_number=63)
        module.setup()
        train_loader = module.train_dataloader()
        data_iter = iter(train_loader)
        image, label, age = next(data_iter)
        # Assuming image size and check shape (batch size, channels, height, width)
        self.assertEqual(
            image.shape, (1, 3, 224, 224)
        )  # Modify according to your image size
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(age, torch.Tensor)


# Create a mock dataset for testing
np.random.seed(42)
data = pd.DataFrame(
    {
        "ID": np.random.choice([f"ID{n}" for n in range(200)], size=10000),
        "CRD": np.random.choice(["Class1", "Class2", "Class3"], size=10000),
        "Feature": np.random.rand(10000),
    }
)


# Function to check class distribution
def check_class_distribution(df, label_col):
    return df[label_col].value_counts(normalize=True)


# Function to check ID uniqueness across datasets
def check_id_uniqueness(train_ids, val_ids, test_ids):
    return len(set(train_ids) & set(val_ids) & set(test_ids)) == 0


class TestStratifiedGroupSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data
        cls.train_val_ids, cls.test_ids = stratified_group_split(
            cls.data, "ID", "CRD", test_size=0.125
        )

        cls.train_val_df = cls.data[cls.data["ID"].isin(cls.train_val_ids)]
        cls.unique_train_val_ids = cls.train_val_df["ID"].unique()
        cls.stratify_train_val_values = (
            cls.train_val_df.groupby("ID")["CRD"].first().values
        )

        cls.train_ids, cls.val_ids = train_test_split(
            cls.unique_train_val_ids,
            test_size=0.142857,
            stratify=cls.stratify_train_val_values,
            random_state=42,
        )

        cls.train_df = cls.data[cls.data["ID"].isin(cls.train_ids)]
        cls.val_df = cls.data[cls.data["ID"].isin(cls.val_ids)]
        cls.test_df = cls.data[cls.data["ID"].isin(cls.test_ids)]

    def test_class_distribution(self):
        train_class_dist = check_class_distribution(self.train_df, "CRD")
        val_class_dist = check_class_distribution(self.val_df, "CRD")
        test_class_dist = check_class_distribution(self.test_df, "CRD")

        self.assertAlmostEqual(
            train_class_dist["Class1"], val_class_dist["Class1"], delta=0.1
        )
        self.assertAlmostEqual(
            train_class_dist["Class2"], val_class_dist["Class2"], delta=0.1
        )
        self.assertAlmostEqual(
            train_class_dist["Class3"], val_class_dist["Class3"], delta=0.1
        )

        self.assertAlmostEqual(
            train_class_dist["Class1"], test_class_dist["Class1"], delta=0.1
        )
        self.assertAlmostEqual(
            train_class_dist["Class2"], test_class_dist["Class2"], delta=0.1
        )
        self.assertAlmostEqual(
            train_class_dist["Class3"], test_class_dist["Class3"], delta=0.1
        )

    def test_id_uniqueness(self):
        id_uniqueness = check_id_uniqueness(self.train_ids, self.val_ids, self.test_ids)
        self.assertTrue(id_uniqueness)


if __name__ == "__main__":
    unittest.main()
