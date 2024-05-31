import torch
from torch import nn
import lightning.pytorch as pl

class EnsembleModel(pl.LightningModule):
    def __init__(self, model_wrappers, num_classes):
        super(EnsembleModel, self).__init__()
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
