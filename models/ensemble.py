import torch
from torch import nn
import lightning.pytorch as pl

class SimpleEnsembleModel(pl.LightningModule):
    def __init__(self, model_wrappers, num_classes):
        super(SimpleEnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_wrappers)
        feature_size = sum([model.output_size for model in model_wrappers])
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        feature_maps = [model.get_feature_map(x) for model in self.models]
        combined_features = torch.cat(feature_maps, dim=1)
        logits = self.fc(combined_features)
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
    def __init__(self, model_wrappers, num_classes):
        super(AdvancedEnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_wrappers)
        feature_size = sum([model.output_size for model in model_wrappers])
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=8)
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        feature_maps = [model.get_feature_map(x) for model in self.models]
        combined_features = torch.cat(feature_maps, dim=1)
        combined_features = combined_features.unsqueeze(0)  # Add sequence dimension
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
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
