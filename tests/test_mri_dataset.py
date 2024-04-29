import unittest
from unittest.mock import patch
import pandas as pd
from dataset import MRIDataModule  # Adjust this import to your actual implementation location

class TestMRIDataset(unittest.TestCase):
    def setUp(self):
        # Mock the json data expected by MRIDataset
        self.mock_json_path = 'path/to/alzheimer_data.json'
        self.mock_data = {
            'class': ['Mild Dementia'] * 3 + ['Non Demented'] * 3 + ['Moderate Dementia'],
            'subject_ID': ['0028', '0028', '0028', '0015', '0015', '0015', '0035'],
            'session': ['1', '2', '3', '1', '2', '3', '1'],
            'mpr': ['1', '1', '1', '1', '1', '1', '1'],
            'paths': [
                [f"Data/Mild Dementia/OAS1_0028_MR1_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Mild Dementia/OAS1_0028_MR2_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Mild Dementia/OAS1_0028_MR3_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Non Demented/OAS1_0015_MR1_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Non Demented/OAS1_0015_MR2_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Non Demented/OAS1_0015_MR3_mpr-1_{i}.jpg" for i in range(100, 161)],
                [f"Data/Moderate Dementia/OAS1_0035_MR1_mpr-1_{i}.jpg" for i in range(100, 161)]
            ]
        }
        self.dataset = MRIDataModule(json_path=self.mock_json_path)

    @patch('pandas.read_json')
    def test_unique_patient_split(self, mock_read_json):
        # Setup mock
        mock_read_json.return_value = pd.DataFrame(self.mock_data)
        
        # Execute
        self.dataset.setup()

        # Check that the same patient ID is not in more than one set
        train_ids = set(self.dataset.train_data['subject_ID'].unique())
        val_ids = set(self.dataset.val_data['subject_ID'].unique())
        test_ids = set(self.dataset.test_data['subject_ID'].unique())
        
        # Assertions
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    @patch('pandas.read_json')
    def test_session_mpr_loading(self, mock_read_json):
        # Setup mock
        mock_read_json.return_value = pd.DataFrame(self.mock_data)
        
        # Execute
        self.dataset.setup()
        dataloader = self.dataset.train_dataloader()
        
        # Check all images in a session are loaded together
        for images, _ in iter(dataloader):
            # Assuming each session+MPR combination should have exactly 61 images (slices from 100 to 160)
            self.assertEqual(len(images), 61)  # Check if 61 slices are loaded

    # Additional tests can be added here for transformations, loading failures, etc.

if __name__ == '__main__':
    unittest.main()
