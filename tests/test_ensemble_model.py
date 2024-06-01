import unittest
import torch
from models import MobileViTWrapper, EfficientNetWrapper, create_ensemble_model

class TestEnsembleModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup that runs once for all tests
        cls.num_labels = 768  # Example feature map size for MobileViT
        cls.num_classes = 4   # Number of output classes
        cls.batch_size = 2
        cls.image_size = (3, 224, 224)  # Example image size (C, H, W)

        # Initialize model wrappers
        cls.mobilevit_wrapper = MobileViTWrapper(model_ckpt='path/to/mobilevit/checkpoint', num_labels=cls.num_labels)
        cls.efficientnet_wrapper = EfficientNetWrapper(model_name='efficientnet-b0', num_classes=cls.num_classes)
        cls.model_wrappers = [cls.mobilevit_wrapper, cls.efficientnet_wrapper]

    def test_simple_ensemble_model_creation(self):
        # Test creation of the simple ensemble model
        simple_model = create_ensemble_model(self.model_wrappers, self.num_classes, use_advanced=False)
        self.assertIsInstance(simple_model, pl.LightningModule)

    def test_advanced_ensemble_model_creation(self):
        # Test creation of the advanced ensemble model
        advanced_model = create_ensemble_model(self.model_wrappers, self.num_classes, use_advanced=True)
        self.assertIsInstance(advanced_model, pl.LightningModule)

    def test_simple_ensemble_model_forward_pass(self):
        # Test forward pass of the simple ensemble model
        simple_model = create_ensemble_model(self.model_wrappers, self.num_classes, use_advanced=False)
        inputs = torch.randn(self.batch_size, *self.image_size)
        outputs = simple_model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes))

    def test_advanced_ensemble_model_forward_pass(self):
        # Test forward pass of the advanced ensemble model
        advanced_model = create_ensemble_model(self.model_wrappers, self.num_classes, use_advanced=True)
        inputs = torch.randn(self.batch_size, *self.image_size)
        outputs = advanced_model(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main()
