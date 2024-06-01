import torch
from torch import nn
import lightning.pytorch as pl

class SimpleEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes):
        super(SimpleEnsembleModel, self).__init__()
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

class AdvancedEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, num_heads=8):
        super(AdvancedEnsembleModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads)
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add sequence dimension for attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)  # Remove sequence dimension
        logits = self.fc(attn_output)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)