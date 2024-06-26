import os
import pandas as pd
import numpy as np
import ast
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import set_plot_style

set_plot_style()

weighted_or_average_f1 = "average"
if weighted_or_average_f1 == "weighted":
    average = "weighted"
    ensemble_column_name = "test_f1_weighted"
    ylabel = "Weighted F1 Score"
elif weighted_or_average_f1 == "average":
    average = "macro"
    ensemble_column_name = "test_f1_individual"
    ylabel = "Average F1 Score"

# Define the root directory of the predictions
root_dir = "predictions"

# Define the slice numbers
slice_numbers = ["65", "86", "56", "95", "62", "35", "59", "74", "80", "134"]


# Function to apply softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Function to calculate weighted F1 score
def calculate_weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=average)


# Dictionary to store F1 scores
f1_scores = {}
f1_scores_merged = {}
f1_scores_single_slice = {}
f1_scores_single_slice_merged = {}


# Function to process CSV files and calculate F1 scores
def process_csv_files(category, model_name, slice_number=None):
    f1_scores[f"{category}_{model_name}"] = []
    f1_scores_merged[f"{category}_{model_name}"] = []

    if slice_number is not None:
        f1_scores_single_slice[f"{category}_{model_name}"] = []
        f1_scores_single_slice_merged[f"{category}_{model_name}"] = []

    for csv_file in os.listdir(model_path):
        if not csv_file.endswith(".csv"):
            continue

        # Load the CSV file
        csv_path = os.path.join(model_path, csv_file)
        df = pd.read_csv(csv_path)

        # Initialize combined predictions
        num_samples = len(df)
        num_classes = len(ast.literal_eval(df[f"slice_{slice_numbers[0]}"].iloc[0]))
        combined_predictions = np.zeros((num_samples, num_classes))

        for slice_number in slice_numbers:
            slice_preds = np.array(
                [
                    softmax(np.array(ast.literal_eval(pred)))
                    for pred in df[f"slice_{slice_number}"]
                ]
            )
            combined_predictions += slice_preds

        # Calculate mean predictions
        mean_predictions = combined_predictions / len(slice_numbers)

        # Get true labels and predicted labels
        y_true = df["labels"].values
        y_pred = np.argmax(mean_predictions, axis=1)

        # Calculate weighted F1 score
        f1 = calculate_weighted_f1(y_true, y_pred)
        f1_scores[f"{category}_{model_name}"].append(f1)

        # Merge classes 1 to 3 into a single class
        y_true_merged = np.where(y_true > 0, 1, y_true)
        y_pred_merged = np.where(y_pred > 0, 1, y_pred)

        # Calculate weighted F1 score for merged classes
        f1_merged = calculate_weighted_f1(y_true_merged, y_pred_merged)
        f1_scores_merged[f"{category}_{model_name}"].append(f1_merged)

        if slice_number is not None:
            slice_preds_single = np.array(
                [
                    softmax(np.array(ast.literal_eval(pred)))
                    for pred in df[f"slice_{slice_number}"]
                ]
            )
            y_pred_single = np.argmax(slice_preds_single, axis=1)
            f1_single = calculate_weighted_f1(y_true, y_pred_single)
            f1_scores_single_slice[f"{category}_{model_name}"].append(f1_single)

            y_pred_single_merged = np.where(y_pred_single > 0, 1, y_pred_single)
            f1_single_merged = calculate_weighted_f1(
                y_true_merged, y_pred_single_merged
            )
            f1_scores_single_slice_merged[f"{category}_{model_name}"].append(
                f1_single_merged
            )


# Iterate over the folders and CSV files
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path):
        continue

    for model_name in os.listdir(category_path):
        model_path = os.path.join(category_path, model_name)
        if not os.path.isdir(model_path):
            continue

        process_csv_files(category, model_name, slice_number="65")

filtered_file_path = "filtered_ensemble_results.csv"
filtered_df = pd.read_csv(filtered_file_path)

# Separate the models
efficientnet_keys = [
    "baseline_models_efficientnet-b2",
    "with_sampler_efficientnet-b2",
    "student_models_efficientnet-b2",
]

mobilevit_keys = [
    "baseline_models_mobilevit-s",
    "with_sampler_mobilevit-s",
    "student_models_mobilevit-s",
]

# Prepare data for plotting
efficientnet_data = [f1_scores_single_slice["baseline_models_efficientnet-b2"]] + [
    f1_scores[key] for key in efficientnet_keys if key in f1_scores
]
mobilevit_data = [f1_scores_single_slice["baseline_models_mobilevit-s"]] + [
    f1_scores[key] for key in mobilevit_keys if key in f1_scores
]

# Add ensemble results to the data
efficientnet_ensemble_data = [
    filtered_df[
        (filtered_df["model_name"] == "efficientnet-b2")
        & (filtered_df["ensemble_variant"] == variant)
    ][ensemble_column_name].values
    for variant in ["simple", "medium", "advanced"]
]
mobilevit_ensemble_data = [
    filtered_df[
        (filtered_df["model_name"] == "MobileVit-s")
        & (filtered_df["ensemble_variant"] == variant)
    ][ensemble_column_name].values
    for variant in ["simple", "medium", "advanced"]
]

efficientnet_data.extend(efficientnet_ensemble_data)
mobilevit_data.extend(mobilevit_ensemble_data)

efficientnet_labels = [
    "Single Slice\nEfficientNet",
    "Previous\n+ 9 Slice",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
    "Simple Ensemble",
    "Medium Ensemble",
    "Advanced Ensemble",
]
mobilevit_labels = [
    "Single Slice\nMobileVit",
    "Previous\n+ 9 Slices",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
    "Simple Ensemble",
    "Medium Ensemble",
    "Advanced Ensemble",
]

efficientnet_means = [np.mean(data) for data in efficientnet_data]
mobilevit_means = [np.mean(data) for data in mobilevit_data]

# Determine y-axis limits
all_f1_scores = efficientnet_data + mobilevit_data
y_min = min([min(scores) for scores in all_f1_scores if len(scores) > 0]) - 0.03
y_max = max([max(scores) for scores in all_f1_scores if len(scores) > 0]) + 0.03

# Create subplots for original classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# Plot EfficientNet models
axes[0].boxplot(efficientnet_data)
axes[0].set_title("F1 Scores for Different EfficientNet-b2 Models")
axes[0].set_xlabel("Model Strategy")
axes[0].set_ylabel(ylabel)
axes[0].set_xticks(range(1, len(efficientnet_labels) + 1))
axes[0].set_xticklabels(efficientnet_labels)
axes[0].set_ylim(y_min, y_max)

for i, mean in enumerate(efficientnet_means):
    axes[0].text(
        i + 1, mean, f"{mean:.2f}", ha="center", va="bottom", color="red", fontsize=12
    )

# Plot MobileViT models
axes[1].boxplot(mobilevit_data)
axes[1].set_title("F1 Scores for Different MobileViT-s Models")
axes[1].set_xlabel("Model Strategy")
axes[1].set_xticks(range(1, len(mobilevit_labels) + 1))
axes[1].set_xticklabels(mobilevit_labels)
axes[1].set_ylim(y_min, y_max)

for i, mean in enumerate(mobilevit_means):
    axes[1].text(
        i + 1, mean, f"{mean:.2f}", ha="center", va="bottom", color="red", fontsize=12
    )

plt.tight_layout()
plt.savefig(
    f"plots/comparative_f1_scores_{average}_boxplot_ensemble_and_model.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
plt.close()
