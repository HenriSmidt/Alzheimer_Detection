import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics.functional import accuracy

class SimpleEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, lr=1e-3):
        super(SimpleEnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.fc = nn.Linear(feature_size, num_classes)
        self.lr = lr

    def forward(self, x):
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MediumEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, lr=1e-3):
        super(SimpleEnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self.lr = lr

    def forward(self, x):
        logits = self.fc(x)
        return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


# Trainer with Early Stopping
early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')
trainer = pl.Trainer(callbacks=[early_stop_callback])


class AdvancedEnsembleModel(pl.LightningModule):
    def __init__(self, feature_size, num_classes, num_heads=8, max_seq_length=10, lr=1e-3):
        super(AdvancedEnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads)
        self.fc = nn.Linear(feature_size, num_classes)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, feature_size))
        self.layer_norm1 = nn.LayerNorm(feature_size)
        self.layer_norm2 = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(p=0.5)
        self.lr = lr

    def forward(self, x):
        x = x.transpose(0, 1)  # Transpose to (sequence_length, batch_size, feature_size)
        
        # Dynamically handle positional encoding length
        seq_length = x.size(0)
        if seq_length > self.positional_encoding.size(0):
            positional_encoding = nn.Parameter(torch.zeros(seq_length, x.size(2))).to(x.device)
            x = x + positional_encoding[:seq_length, :].unsqueeze(1)
        else:
            x = x + self.positional_encoding[:seq_length, :].unsqueeze(1)
        
        # Apply layer normalization before attention
        x = self.layer_norm1(x)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Apply layer normalization after attention
        attn_output = self.layer_norm2(attn_output)
        
        # Apply dropout after attention
        attn_output = self.dropout(attn_output)
        
        # We take the mean of the attention outputs across the sequence dimension
        attn_output = attn_output.mean(dim=0)  # Now shape is (batch_size, feature_size)
        
        # Pass the result through the fully connected layer
        output = self.fc(attn_output)
        
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
