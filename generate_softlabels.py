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
from utils import get_best_device, set_reproducibility


def generate_soft_labels(model, dataloader, device, output_dir, slice_number):
    model.eval()
    soft_labels = {}

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device).float()
            ids = batch["id"]
            logits = model(inputs).cpu().numpy()  # Save raw logits

            for id, logit in zip(ids, logits):
                soft_labels[id] = logit

    # Save soft labels for each slice
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"soft_labels_slice_{slice_number}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(soft_labels, f)
    print(f"Saved soft labels to {output_file}")


def get_ckpt_files(folder_path):
    # Returns all ckpt files in the folder
    return glob.glob(os.path.join(folder_path, "*.ckpt"))


def main():
    set_reproducibility(42)  # So that the train val test split remains the same

    # Paths and parameters
    csv_path = "Data/metadata_for_preprocessed_files.csv"
    batch_size = 32
    num_workers = 0
    device = get_best_device()

    # Base directory for model checkpoints
    base_checkpoint_dir = "model_checkpoints/with_custom_sampler"

    # Model configurations
    model_names = ["efficientnet-b2", "mobilevit-s"]

    for model_name in model_names:
        checkpoint_dir = os.path.join(base_checkpoint_dir, model_name)
        checkpoint_files = get_ckpt_files(checkpoint_dir)

        slice_model_dict = {}
        for ckpt_path in checkpoint_files:
            filename = os.path.basename(ckpt_path)
            slice_number = int(filename.split("_")[2])
            slice_model_dict[ckpt_path] = slice_number

        for ckpt_path, slice_number in slice_model_dict.items():
            # Load the appropriate model
            if "efficientnet" in model_name:
                model = EfficientNetBaseline.load_from_checkpoint(
                    ckpt_path, model_name=model_name, num_classes=4
                )
                transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            elif "MobileVit" in model_name:
                model_ckpt = "apple/mobilevit-small"
                model = MobileViTLightning.load_from_checkpoint(
                    ckpt_path, model_ckpt=model_ckpt, num_labels=4
                )
                processor = MobileViTImageProcessor.from_pretrained(model_ckpt)

                def transform(image):
                    # Use MobileViTImageProcessor for preprocessing
                    return processor(image, return_tensors="pt")[
                        "pixel_values"
                    ].squeeze(0)

            model.to(device)

            data_module = MRIImageDataModule(
                csv_path,
                slice_number=slice_number,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
                always_return_id=True,
            )
            data_module.setup()
            dataloader = data_module.train_dataloader(shuffle=False)

            # Output directory for soft labels
            output_dir = f"soft_labels/custom_sampler/{model_name}"
            generate_soft_labels(model, dataloader, device, output_dir, slice_number)


if __name__ == "__main__":
    main()
