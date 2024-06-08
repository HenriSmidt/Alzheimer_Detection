import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from lightning.pytorch import LightningModule

class EfficientNetBaseline(LightningModule):
    def __init__(self, model_name='efficientnet-b0', num_classes=4, lr=1e-3, alpha=0.5, temperature=2.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNet.from_pretrained(model_name)
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.kd_criterion = nn.KLDivLoss(reduction='batchmean')
        self.lr = lr
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        return self.model.extract_features(x)

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['labels']
        soft_labels = batch.get('soft_labels', None)
        
        inputs = inputs.float()
        logits = self(inputs)
        loss_ce = self.criterion(logits, labels)

        if soft_labels is not None:
            soft_labels = soft_labels.to(self.device)
            # Apply temperature scaling
            logits_distilled = torch.nn.functional.log_softmax(logits / self.temperature, dim=1)
            soft_labels_distilled = torch.nn.functional.softmax(soft_labels / self.temperature, dim=1)
            # Compute knowledge distillation loss
            loss_kd = self.kd_criterion(logits_distilled, soft_labels_distilled)
            # Combine the two losses
            loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd * (self.temperature ** 2)
        else:
            loss = loss_ce

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['labels']

        inputs = inputs.float()
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
