import pytorch_lightning as pl
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
import torch

class MobileViTLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-x-small")
        self.feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-x-small")
    
    def forward(self, x):
        outputs = self.model(**x)
        return outputs.logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        logits = self.forward(inputs)
        loss = self.loss(logits, labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels)
