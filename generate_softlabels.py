import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MRIImageDataModule
from models import MobileViTLightning, EfficientNetBaseline
from transformers import MobileViTImageProcessor
import pickle
import glob
from utils import get_best_device

def generate_soft_labels(model, dataloader, device, output_dir, slice_number):
    model.eval()
    soft_labels = {}
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['inputs'].to(device).float()
            ids = batch['id']
            logits = model(inputs)
            soft_labels_batch = torch.nn.functional.softmax(logits / model.temperature, dim=1).cpu().numpy()
            
            for id, soft_label in zip(ids, soft_labels_batch):
                soft_labels[id] = soft_label

    # Save soft labels for each slice
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"soft_labels_slice_{slice_number}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(soft_labels, f)
    print(f"Saved soft labels to {output_file}")

def get_checkpoint_files(base_dir, model_name):
    # Construct the glob pattern to retrieve all checkpoint files
    pattern = os.path.join(base_dir, model_name, "*.ckpt")
    return glob.glob(pattern)

def main():
    # Paths and parameters
    csv_path = "Data/metadata_for_preprocessed_files.csv"
    batch_size = 32
    num_workers = 0
    slices = [35, 62, 63, 64]  # Example slice numbers, adjust as needed
    device = get_best_device()

    # Base directory for model checkpoints
    base_checkpoint_dir = "model_checkpoints/with_scheduler"
    
    # Model configurations
    model_names = ["efficientnet-b2", "MobileVit"]

    for model_name in model_names:
        checkpoint_files = get_checkpoint_files(base_checkpoint_dir, model_name)
        
        for ckpt_path in checkpoint_files:
            # Load the appropriate model
            if "efficientnet" in model_name:
                model = EfficientNetBaseline.load_from_checkpoint(ckpt_path, model_name=model_name, num_classes=4)
            elif "MobileVit" in model_name:
                model_ckpt = "apple/mobilevit-small"
                model = MobileViTLightning.load_from_checkpoint(ckpt_path, model_ckpt=model_ckpt, num_labels=4)
                processor = MobileViTImageProcessor.from_pretrained(model_ckpt)
                def transform(image):
                    # Use MobileViTImageProcessor for preprocessing
                    return processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
            
            model.to(device)

            for slice_number in slices:
                # Prepare the data module and dataloader for the current slice
                data_module = MRIImageDataModule(
                    csv_path,
                    slice_number=slice_number,
                    transform=transform,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    always_return_id=True
                )
                data_module.setup()
                dataloader = data_module.train_dataloader()

                # Output directory for soft labels
                output_dir = f"soft_labels/{model_name}/{os.path.basename(ckpt_path).replace('.ckpt', '')}"
                generate_soft_labels(model, dataloader, device, output_dir, slice_number)

if __name__ == "__main__":
    main()
