import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam

class AlzheimerEfficientNet(LightningModule):
    def __init__(self, num_classes=4, lr=1e-3):
        super().__init__()
        # Load a pre-trained EfficientNet
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # Replace the classifier layer with the correct number of outputs for our task
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        # x is [N, C, H, W]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # Flatten the batch dimension and the slice dimension
        images = images.view(-1, 3, 224, 224)  # Adjust dimensions for EfficientNet input
        labels = labels.repeat_interleave(images.size(0) // labels.size(0))
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(-1, 3, 224, 224)
        labels = labels.repeat_interleave(images.size(0) // labels.size(0))
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        images = images.view(-1, 3, 224, 224)
        labels = labels.repeat_interleave(images.size(0) // labels.size(0))
        logits = self(images)
        # Compute softmax probabilities for aggregation
        probs = torch.softmax(logits, dim=1)
        # Aggregate probabilities by mean to produce a single prediction per sample
        probs = probs.view(labels.size(0), -1, probs.shape[1]).mean(dim=1)
        loss = self.criterion(probs, labels)
        self.log('test_loss', loss)
        return {'loss': loss, 'probs': probs, 'labels': labels}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

# Example use case:
# dataset and dataloader setup must be done according to your actual setup
# model = AlzheimerEfficientNet()
# trainer = Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader, val_dataloader)
