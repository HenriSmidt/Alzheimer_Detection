import lightning.pytorch as pl
from transformers import MobileViTForImageClassification
import torch

class MobileViTLightning(pl.LightningModule):
    def __init__(self, model_ckpt, num_labels, lr=2e-5, self_distillation_alpha=0.5, self_distillation_temperature=2.0):
        super(MobileViTLightning, self).__init__()
        self.model = MobileViTForImageClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.kd_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.lr = lr
        self.self_distillation_alpha = self_distillation_alpha
        self.self_distillation_temperature = self_distillation_temperature

    def forward(self, x):
        return self.model(x).logits

    def extract_features(self, x):
        outputs = self.model(x, output_hidden_states=True)
        return outputs.hidden_states[-1]

    def training_step(self, batch, batch_idx):
        inputs = batch['inputs']
        labels = batch['labels']
        soft_labels = batch.get('soft_labels', None)
        
        inputs = inputs.float()
        logits = self(inputs)
        loss_ce = self.criterion(logits, labels)

        if soft_labels is not None:
            soft_labels = soft_labels.to(self.device)
            # Apply self_distillation_temperature scaling
            logits_distilled = torch.nn.functional.log_softmax(logits / self.self_distillation_temperature, dim=1)
            soft_labels_distilled = torch.nn.functional.softmax(soft_labels / self.self_distillation_temperature, dim=1)
            # Compute knowledge distillation loss
            loss_kd = self.kd_criterion(logits_distilled, soft_labels_distilled)
            # Combine the two losses
            loss = self.self_distillation_alpha * loss_ce + (1 - self.self_distillation_alpha) * loss_kd * (self.self_distillation_temperature ** 2)
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
