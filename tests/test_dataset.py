import unittest
import pandas as pd
from torchvision import transforms
import sys
import os
import torch

# Add the parent directory to sys.path to import dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import MRIImageDataModule, MRIDataset

# Your dataset and data module classes here (MRIDataset, MRIImageDataModule)
# Make sure to import or define them if they are in another file.

class TestMRIDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV for testing
        self.temp_csv_path = 'temp_test_data.csv'
        data = {
            'ID': ['OAS1_0001_MR1', 'OAS1_0001_MR1', 'OAS1_0001_MR1', 'OAS1_0002_MR1', 'OAS1_0002_MR1', 'OAS1_0003_MR1'],
            'slice_number': [62, 63, 64, 63, 64, 63],
            'path': [
                'Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_62.jpeg',
                'Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg',
                'Data/OASIS_Extracted/OAS1_0001/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg',
                'Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg',
                'Data/OASIS_Extracted/OAS1_0002/OAS1_0002_MR1_mpr_n4_anon_111_t88_gfc_slice_64.jpeg',
                'Data/OASIS_Extracted/OAS1_0003/OAS1_0003_MR1_mpr_n4_anon_111_t88_gfc_slice_63.jpeg'
            ],
            'CDR': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'is_masked': [True, False, True, False, True, False],
            'Age': [44, 55, 66, 77, 88, 99]

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
        self.assertEqual(len(module.train_dataset)+len(module.test_dataset), 3)  # Should have unique ID entries for training

    def test_data_loading(self):
        module = MRIImageDataModule(self.temp_csv_path, batch_size=1, slice_number=63)
        module.setup()
        train_loader = module.train_dataloader()
        data_iter = iter(train_loader)
        image, label, age = next(data_iter)
        # Assuming image size and check shape (batch size, channels, height, width)
        self.assertEqual(image.shape, (1, 3, 224, 224))  # Modify according to your image size
        self.assertIsInstance(label, torch.Tensor)
        self.assertIsInstance(age, torch.Tensor)
        
if __name__ == '__main__':
    unittest.main()
