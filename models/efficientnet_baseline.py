import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam

class EfficientNetBaseline(LightningModule):
    def __init__(self, model_name='efficientnet-b0', num_classes=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  
        # Load a pre-trained EfficientNet
        self.model = EfficientNet.from_pretrained(model_name)
        # Replace the classifier layer with the correct number of outputs for our task
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x, return_features=False):
        # x is [N, C, H, W]
        if return_features:
            features = self.model.extract_features(x)
            return features
        else:
            return self.model(x)

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
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_feature_map(self, x):
        with torch.no_grad():
            self.eval()
            features = self(x, return_features=True)
            return features

# Example usage:
# dataset and dataloader setup must be done according to your actual setup
# model = EfficientNetBaseline()
# trainer = Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader, val_dataloader)

# To get the feature map for a batch of images:
# images, labels, age = next(iter(train_dataloader))
# features = model.get_feature_map(images)
# print(features.shape)  # Shape will depend on the EfficientNet version used
