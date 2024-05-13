import unittest
import torch
from models import MobileViTLightning 

class TestMobileViTLightning(unittest.TestCase):

    def setUp(self):
        # This method is called before each test
        self.model = MobileViTLightning()

    def test_initialization(self):
        # Test whether the model and its components are properly initialized
        self.assertIsNotNone(self.model.model, "The underlying MobileViT model should not be None.")
        self.assertIsNotNone(self.model.feature_extractor, "The feature extractor should not be None.")

    def test_forward_pass(self):
        # Create a dummy image tensor (e.g., 1 sample, 3 channels, 224x224 pixels)
        # Adjust size according to what MobileViT expects
        dummy_image = torch.rand(1, 3, 224, 224)
        inputs = self.model.feature_extractor(images=dummy_image, return_tensors="pt")

        # Execute forward pass
        with torch.no_grad():  # Use no_grad to avoid tracking gradients
            logits = self.model.forward(inputs)

        # Check if the output has the expected shape (assuming 1000 classes from ImageNet)
        self.assertEqual(logits.shape, torch.Size([1, 1000]), "Output logits should have shape [1, 1000]")

# Add more tests as needed to cover other functionalities like the training step, validation step, etc.

if __name__ == '__main__':
    unittest.main()
