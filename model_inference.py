import torch
import pandas as pd
import numpy as np
from transformers import MobileViTImageProcessor
import os
from dataset import MRIImageDataModule
from models import MobileViTLightning, EfficientNetBaseline
from utils import get_best_device
from torchvision import transforms
from tqdm import tqdm

# Set device
device = get_best_device()

# Path to the CSV file
csv_path = "Data/metadata_for_preprocessed_files.csv"

# Define the models
model_configs = {
    "baseline_models": {
        "efficientnet-b0": {
            "model_ckpt": None,
        },
        "efficientnet-b2": {
            "model_ckpt": None,
        },
        "mobilevit-s": {
            "model_ckpt": "apple/mobilevit-small",
        },
    },
    "with_sampler": {
        "efficientnet-b2": {
            "model_ckpt": None,
        },
        "mobilevit-s": {
            "model_ckpt": "apple/mobilevit-small",
        },
    },
    "student_models": {
        "efficientnet-b2": {
            "model_ckpt": None,
        },
        "mobilevit-s": {
            "model_ckpt": "apple/mobilevit-small",
        },
    },
}

# Define the slice numbers
slice_numbers = ["65", "86", "56", "95", "62", "35", "59", "74", "80", "134"]

# Load and preprocess the MRI dataset
def get_transform(model_name, model_ckpt):
    if model_name == "mobilevit-s":
        processor = MobileViTImageProcessor.from_pretrained(model_ckpt)
        return lambda image: processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
    else:  # For efficientnet models
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform

# Function to apply softmax after averaging predictions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Function to run inference
def run_inference_and_save_predictions(model_configs, device, csv_path, runs=10):
    for category, models in model_configs.items():
        for model_name, config in models.items():
            model_ckpt = config["model_ckpt"]
            transform = get_transform(model_name, model_ckpt)
            
            for run in range(runs):
                results_dict = {"id": [], "labels": []}
                model_predictions = {f"slice_{slice_number}": [] for slice_number in slice_numbers}

                for slice_number in slice_numbers:
                    # Find the correct model checkpoint path
                    folder_path = f"model_checkpoints/{category}/{model_name}/"
                    model_path = None
                    for filename in os.listdir(folder_path):
                        if filename.startswith(f"slice_number_{slice_number}") and filename.endswith(".ckpt"):
                            model_path = os.path.join(folder_path, filename)
                            break

                    if not model_path:
                        print(f'The checkpoint path for slice {slice_number} does not exist in {folder_path}')
                        continue

                    # Load the model
                    if model_name == "mobilevit-s":
                        model = MobileViTLightning.load_from_checkpoint(
                            model_path, model_ckpt=model_ckpt, num_labels=4
                        )
                    else:
                        model = EfficientNetBaseline.load_from_checkpoint(
                            model_path, model_name=model_name, num_classes=4
                        )

                    model = model.to(device)
                    model.eval()

                    # Initialize the data module
                    data_module = MRIImageDataModule(
                        csv_path,
                        slice_number=int(slice_number),
                        transform=transform,
                        batch_size=40,
                        num_workers=0,
                    )
                    data_module.setup()
                    test_loader = data_module.test_dataloader()

                    # Perform inference and store predictions
                    all_preds = []
                    all_labels = []
                    all_ids = []

                    with torch.no_grad():
                        for batch in tqdm(test_loader, desc=f"Inference {category}/{model_name} slice {slice_number} run {run+1}/{runs}"):
                            inputs = batch["inputs"]
                            labels = batch["labels"]
                            ids = batch["id"]
                            inputs = inputs.to(device).float()
                            outputs = model(inputs)
                            all_preds.extend(outputs.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                            all_ids.extend(ids)

                    # Store predictions for this slice
                    model_predictions[f"slice_{slice_number}"] = all_preds
                    if not results_dict["id"]:
                        results_dict["id"] = all_ids
                        results_dict["labels"] = all_labels

                # Aggregate and average predictions
                combined_predictions = np.zeros_like(model_predictions[f"slice_{slice_numbers[0]}"])
                for slice_number in slice_numbers:
                    slice_preds = np.array(model_predictions[f"slice_{slice_number}"])
                    combined_predictions += slice_preds
                    results_dict[f"slice_{slice_number}"] = slice_preds.tolist()

                averaged_predictions = combined_predictions / len(slice_numbers)
                final_predictions = softmax(averaged_predictions)  # Apply softmax

                results_dict[f"mean_predictions"] = final_predictions.tolist()

                # Convert dictionary to DataFrame
                df_results = pd.DataFrame(results_dict)

                # Ensure rows are sorted by MRI ID
                df_results = df_results.sort_values(by="id")

                # Create the output folder structure if it doesn't exist
                output_folder = f"predictions/{category}/{model_name}"
                os.makedirs(output_folder, exist_ok=True)

                # Save DataFrame to CSV
                df_results.to_csv(f"{output_folder}/run_{run}_predictions.csv", index=False)

                # Display DataFrame
                print(df_results.head())

run_inference_and_save_predictions(model_configs, device, csv_path)
