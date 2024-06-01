import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from lightning.pytorch import LightningModule

class EfficientNetBaseline(LightningModule):
    def __init__(self, model_name='efficientnet-b0', num_classes=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNet.from_pretrained(model_name)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        return self.model.extract_features(x)

    def training_step(self, batch, batch_idx):
        images, labels, age = batch
        images = images.float()
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, age = batch
        images = images.float()
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
