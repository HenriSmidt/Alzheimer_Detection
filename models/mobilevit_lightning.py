import lightning.pytorch as pl
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torch

class MobileViTLightning(pl.LightningModule):
    def __init__(self, model_ckpt, num_labels):
        super(MobileViTLightning, self).__init__()
        self.model = MobileViTForImageClassification.from_pretrained(model_ckpt, num_labels=num_labels, ignore_mismatched_sizes=True)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
