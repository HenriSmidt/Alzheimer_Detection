import torch
from torch import nn
import lightning.pytorch as pl

class SimpleEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, lr=1e-3):
        super(SimpleEnsembleModel, self).__init__()
        self.fc = nn.Linear(feature_size, num_classes)
        self.lr = lr

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
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class AdvancedEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, num_heads=8, max_seq_length=10, lr = 1e-3):
        super(AdvancedEnsembleModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads)
        self.fc = nn.Linear(feature_size, num_classes)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, feature_size))
        self.layer_norm1 = nn.LayerNorm(feature_size)
        self.layer_norm2 = nn.LayerNorm(feature_size)
        self.lr = lr

    def forward(self, x):
        # x has shape (batch_size, sequence_length, feature_size)
        x = x.transpose(0, 1)  # Transpose to (sequence_length, batch_size, feature_size)
        
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(0), :].unsqueeze(1)
        
        # Apply layer normalization before attention
        x = self.layer_norm1(x)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Apply layer normalization after attention
        attn_output = self.layer_norm2(attn_output)
        
        # We take the mean of the attention outputs across the sequence dimension
        attn_output = attn_output.mean(dim=0)  # Now shape is (batch_size, feature_size)
        
        # Pass the result through the fully connected layer
        output = self.fc(attn_output)
        
        return output

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
        return torch.optim.AdamW(self.parameters(), self.lr)
