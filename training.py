import torch
from torchvision import transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import MobileViTImageProcessor
import wandb

from dataset import MRIImageDataModule, MRIDataset
from models import MobileViTLightning
from utils import get_best_device, LoginCredentials

from datetime import datetime
import lightning.pytorch as pl
import torch
import numpy as np
import random
from sklearn.metrics import f1_score

authenticator = LoginCredentials()
wandb.login(key=authenticator.wandb_key)

def set_reproducibility(seed=42):
    # Set Python random seed
    random.seed(seed)
    
    # Set Numpy seed
    np.random.seed(seed)
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Control sources of nondeterminism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch Lightning utility to seed everything
    pl.seed_everything(seed, workers=True)

# Example of setting up a reproducible environment
set_reproducibility(42)

# Load the preprocessor
model_ckpt = "apple/mobilevit-x-small"
processor = MobileViTImageProcessor.from_pretrained(model_ckpt)

# Load and preprocess the CIFAR-10 dataset
def transform(image):
    # Use MobileViTImageProcessor for preprocessing
    return processor(image, return_tensors="pt")["pixel_values"].squeeze(0)

csv_path = 'Data/metadata_for_preprocessed_files.csv'

# Define sweep configuration
sweep_config = {
    'method': 'grid',
    'parameters': {
        'slice_number': {
            'values': list(range(20, 140, 3))  # Values: 20, 17, 14
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="Alzheimer-Detection")

# Define the training function
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        data_module = MRIImageDataModule(csv_path, slice_number=config.slice_number, transform=transform, batch_size=64)
        data_module.setup()
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        test_loader = data_module.test_dataloader()

        model = MobileViTLightning(model_ckpt=model_ckpt, num_labels=4)

        wandb_logger = WandbLogger()

        checkpoint_callback = ModelCheckpoint(
            dirpath='model_checkpoints',
            filename=f'slice_numer_{config.slice_number}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )

        trainer = L.Trainer(
            max_epochs=20,
            devices='auto',
            accelerator='auto',
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Load best model for testing
        # best_model_path = checkpoint_callback.best_model_path
        # best_model = MobileViTLightning.load_from_checkpoint(best_model_path, model_ckpt=model_ckpt, num_labels=4)

        # # Evaluate on test set
        # best_model.eval()
        # all_preds = []
        # all_labels = []
        # for batch in test_loader:
        #     inputs, labels = batch
        #     outputs = best_model(inputs)
        #     preds = torch.argmax(outputs, dim=1)
        #     all_preds.extend(preds.cpu().numpy())
        #     all_labels.extend(labels.cpu().numpy())

        # f1 = f1_score(all_labels, all_preds, average='weighted')
        # wandb.log({'test_f1_score': f1})

# Run the sweep
wandb.agent(sweep_id, function=train)
