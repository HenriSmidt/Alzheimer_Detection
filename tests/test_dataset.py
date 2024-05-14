import unittest
import pandas as pd
from torchvision import transforms
from dataset import MRIDataset, MRIImageDataModule
def setUp(self):
    # Sample data mimicking the actual dataset structure
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
        'CDR': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Mimic reading the DataFrame as done in the actual data module setup
    self.data_module = MRIImageDataModule(data_path='Data/metadata_for_preprocessed_files.csv', batch_size=2, slice_number=63)
    # Instead of loading from a file, directly use the DataFrame created above
    train_ids, test_ids = train_test_split(df['ID'].unique(), test_size=0.2, random_state=42)
    train_df = df[df['ID'].isin(train_ids)]
    test_df = df[df['ID'].isin(test_ids)]

    # Assign these DataFrames to the data module's datasets
    self.data_module.train_dataset = MRIDataset(train_df, self.data_module.slice_number, transform=self.transform)
    self.data_module.test_dataset = MRIDataset(test_df, self.data_module.slice_number, transform=self.transform)

def test_train_test_id_separation(self):
    train_ids = set(self.data_module.train_dataset.df.index.get_level_values(0))
    test_ids = set(self.data_module.test_dataset.df.index.get_level_values(0))
    self.assertTrue(train_ids.isdisjoint(test_ids))

def test_dataloader_output(self):
    train_loader = self.data_module.train_dataloader()
    for batch in train_loader:
        # Each batch should have the dimensions [batch_size, channels, height, width]
        self.assertEqual(batch.shape, (2, 3, 256, 256))  # Update dimensions based on actual image sizes
        break
