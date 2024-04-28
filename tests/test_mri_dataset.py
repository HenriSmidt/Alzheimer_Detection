import unittest
from dataset import MRIDataset  # Assuming your loader class is in mri_loader.py
import os

class TestMRIDataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = "Data"  # Specify the path to your data directory
        self.dataset = MRIDataset(root_dir=self.data_dir)
        self.train_loader, self.test_loader = self.dataset.get_data_loaders(test_size=0.2, batch_size=1, random_seed=42)

    def test_patient_split(self):
        train_patients = set()
        test_patients = set()

        # Collect all patient IDs in the train loader
        for images, _ in self.train_loader:
            for img_batch in images:
                patient_id = os.path.basename(img_batch[0].filename).split('_')[1]
                train_patients.add(patient_id)

        # Collect all patient IDs in the test loader
        for images, _ in self.test_loader:
            for img_batch in images:
                patient_id = os.path.basename(img_batch[0].filename).split('_')[1]
                test_patients.add(patient_id)

        # Check that no patient ID is in both training and test sets
        self.assertTrue(train_patients.isdisjoint(test_patients))

    def test_session_mpr_grouping(self):
        # Test that all images in a batch are from the same session and MPR
        for loader in (self.train_loader, self.test_loader):
            for images, _ in loader:
                for img_batch in images:
                    session_mpr_ids = set()
                    for img in img_batch:
                        session_mpr_id = '_'.join(os.path.basename(img.filename).split('_')[2:4])
                        session_mpr_ids.add(session_mpr_id)

                    # Check if all images in the batch have the same session_mpr_id
                    self.assertEqual(len(session_mpr_ids), 1, "Images from different sessions or MPRs were grouped together.")

if __name__ == "__main__":
    unittest.main()
