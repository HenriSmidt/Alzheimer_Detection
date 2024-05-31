import lightning.pytorch as pl
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torch


class MobileViTLightning(pl.LightningModule):
    def __init__(self, model_ckpt, num_labels):
        super(MobileViTLightning, self).__init__()
        self.model = MobileViTForImageClassification.from_pretrained(
            model_ckpt, num_labels=num_labels, ignore_mismatched_sizes=True
        )

    def forward(self, x, return_features=False):
        if return_features:
            outputs = self.model(x, output_hidden_states=True)
            return outputs.hidden_states[-1]  # Get the last hidden state
        else:
            return self.model(x).logits

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

    def get_feature_map(self, x):
        with torch.no_grad():
            self.eval()
            features = self(x, return_features=True)
            return features

# Example usage:
# dataset and dataloader setup must be done according to your actual setup
# model = MobileViTLightning(model_ckpt='path/to/checkpoint', num_labels=4)
# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, train_dataloader, val_dataloader)

# To get the feature map for a batch of images:
# images, labels, _ = next(iter(train_dataloader))
# features = model.get_feature_map(images)
# print(features.shape)  # Shape will depend on the model configuration
