import os
import pandas as pd
import numpy as np
import ast
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm
from utils import set_plot_style

# Settings for the f1 calculation
weighted_or_average_f1 = "macro"
if weighted_or_average_f1 == "weighted":
    average = "weighted"
    ylabel = "Weighted F1 Score"
elif weighted_or_average_f1 == "macro":
    average = "macro"
    ylabel = "Average F1 Score"


set_plot_style()

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

# Separate the models
efficientnet_keys = [
    "baseline_models_efficientnet-b2",
    "with_custom_sampler_efficientnet-b2",
    "student_models_with_custom_sampler_efficientnet-b2",
]

mobilevit_keys = [
    "baseline_models_mobilevit-s",
    "with_sampler_mobilevit-s",
    "student_models_with_custom_sampler_mobilevit-s",
]

# Prepare data for plotting
efficientnet_data = [f1_scores_single_slice["baseline_models_efficientnet-b2"]] + [
    f1_scores[key] for key in efficientnet_keys if key in f1_scores
]
mobilevit_data = [f1_scores_single_slice["baseline_models_mobilevit-s"]] + [
    f1_scores[key] for key in mobilevit_keys if key in f1_scores
]

efficientnet_data_merged = [
    f1_scores_single_slice_merged["baseline_models_efficientnet-b2"]
] + [f1_scores_merged[key] for key in efficientnet_keys if key in f1_scores_merged]
mobilevit_data_merged = [
    f1_scores_single_slice_merged["baseline_models_mobilevit-s"]
] + [f1_scores_merged[key] for key in mobilevit_keys if key in f1_scores_merged]

efficientnet_labels = [
    "Single Slice\nEfficientNet",
    "Previous\n+ 9 Slice",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
]
mobilevit_labels = [
    "Single Slice\nMobileVit",
    "Previous\n+ 9 Slices",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
]

efficientnet_means = [
    np.mean(f1_scores_single_slice["baseline_models_efficientnet-b2"])
] + [np.mean(f1_scores[key]) for key in efficientnet_keys if key in f1_scores]
mobilevit_means = [np.mean(f1_scores_single_slice["baseline_models_mobilevit-s"])] + [
    np.mean(f1_scores[key]) for key in mobilevit_keys if key in f1_scores
]

efficientnet_means_merged = [
    np.mean(f1_scores_single_slice_merged["baseline_models_efficientnet-b2"])
] + [
    np.mean(f1_scores_merged[key])
    for key in efficientnet_keys
    if key in f1_scores_merged
]
mobilevit_means_merged = [
    np.mean(f1_scores_single_slice_merged["baseline_models_mobilevit-s"])
] + [
    np.mean(f1_scores_merged[key]) for key in mobilevit_keys if key in f1_scores_merged
]

# Determine y-axis limits
all_f1_scores = efficientnet_data + mobilevit_data
y_min = min([min(scores) for scores in all_f1_scores]) - 0.03
y_max = max([max(scores) for scores in all_f1_scores]) + 0.03

all_f1_scores_merged = efficientnet_data_merged + mobilevit_data_merged
y_min_merged = min([min(scores) for scores in all_f1_scores_merged]) - 0.03
y_max_merged = max([max(scores) for scores in all_f1_scores_merged]) + 0.03

# Create subplots for original classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)

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
    f"plots/custom_comparative_f1_scores_{average}_boxplot.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
plt.close()

# Create subplots for merged classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)

# Plot EfficientNet models with merged classes
axes[0].boxplot(efficientnet_data_merged)
axes[0].set_title("F1 Scores for EfficientNet-b2 Models (Classes Merged)")
axes[0].set_xlabel("Model Strategy")
axes[0].set_ylabel(ylabel)
axes[0].set_xticks(range(1, len(efficientnet_labels) + 1))
axes[0].set_xticklabels(efficientnet_labels)
axes[0].set_ylim(y_min_merged, y_max_merged)

for i, mean in enumerate(efficientnet_means_merged):
    axes[0].text(
        i + 1, mean, f"{mean:.2f}", ha="center", va="bottom", color="red", fontsize=12
    )

# Plot MobileViT models with merged classes
axes[1].boxplot(mobilevit_data_merged)
axes[1].set_title("F1 Scores for MobileViT-s Models (Classes Merged)")
axes[1].set_xlabel("Model Strategy")
axes[1].set_xticks(range(1, len(mobilevit_labels) + 1))
axes[1].set_xticklabels(mobilevit_labels)
axes[1].set_ylim(y_min_merged, y_max_merged)

for i, mean in enumerate(mobilevit_means_merged):
    axes[1].text(
        i + 1, mean, f"{mean:.2f}", ha="center", va="bottom", color="red", fontsize=12
    )

plt.tight_layout()
plt.savefig(
    f"plots/custom_comparative_f1_scores_{average}_boxplot_merged.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
plt.close()


set_plot_style()
class_labels = [0, 1, 2]  # These are supposed to be the class labels per strategy
custom_class_labels = ["0", "0.5", "1"]  # Custom class labels

# Define the root directory of the predictions
root_dir = "predictions"

# Define the slice numbers
slice_numbers = ["65", "86", "56", "95", "62", "35", "59", "74", "80", "134"]


# Function to apply softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Dictionary to store F1 scores
f1_scores = {}
f1_scores_single_slice = {}


# Function to process CSV files and calculate F1 scores
def process_csv_files(category, model_name, slice_number=None):
    f1_scores[f"{category}_{model_name}"] = {
        class_label: [] for class_label in class_labels
    }

    if slice_number is not None:
        f1_scores_single_slice[f"{category}_{model_name}"] = {
            class_label: [] for class_label in class_labels
        }

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

        # Calculate class-wise F1 scores
        class_f1_scores = f1_score(y_true, y_pred, average=None, labels=class_labels)
        for i, class_label in enumerate(class_labels):
            f1_scores[f"{category}_{model_name}"][class_label].append(
                class_f1_scores[i]
            )

        if slice_number is not None:
            slice_preds_single = np.array(
                [
                    softmax(np.array(ast.literal_eval(pred)))
                    for pred in df[f"slice_{slice_number}"]
                ]
            )
            y_pred_single = np.argmax(slice_preds_single, axis=1)
            class_f1_scores_single = f1_score(
                y_true, y_pred_single, average=None, labels=class_labels
            )
            for i, class_label in enumerate(class_labels):
                f1_scores_single_slice[f"{category}_{model_name}"][class_label].append(
                    class_f1_scores_single[i]
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

# Separate the models
efficientnet_keys = [
    "baseline_models_efficientnet-b2",
    "with_custom_sampler_efficientnet-b2",
    "student_models_with_custom_sampler_efficientnet-b2",
]

mobilevit_keys = [
    "baseline_models_mobilevit-s",
    "with_sampler_mobilevit-s",
    "student_models_with_custom_sampler_mobilevit-s",
]


# Prepare data for plotting
def prepare_data(keys, single_slice_key):
    data = []
    for class_label in class_labels:
        data.append(f1_scores_single_slice[single_slice_key][class_label])
    for key in keys:
        if key in f1_scores:
            for class_label in class_labels:
                data.append(f1_scores[key][class_label])
    return data


efficientnet_data = prepare_data(efficientnet_keys, "baseline_models_efficientnet-b2")
mobilevit_data = prepare_data(mobilevit_keys, "baseline_models_mobilevit-s")

efficientnet_labels = [
    "Single Slice\nEfficientNet",
    "Previous\n+ 9 Slices",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
]
mobilevit_labels = [
    "Single Slice\nMobileVit",
    "Previous\n+ 9 Slices",
    "Previous\n+ Sampler",
    "Previous\n+ Self-Distillation",
]

# Determine y-axis limits
all_f1_scores = efficientnet_data + mobilevit_data
y_min = min([min(scores) for scores in all_f1_scores if len(scores) > 0]) - 0.03
y_max = max([max(scores) for scores in all_f1_scores if len(scores) > 0]) + 0.03

# Create subplots for original classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharey=True)


# Function to plot data
def plot_data(ax, data, labels, title, ylabel=None):
    positions = []
    for i in range(len(labels)):
        positions.extend(
            range(
                i * len(custom_class_labels) + 1, (i + 1) * len(custom_class_labels) + 1
            )
        )

    flattened_positions = positions
    for idx in range(len(labels)):
        for class_label in class_labels:
            pos_offset = idx * len(custom_class_labels) + class_label
            class_data = data[pos_offset]
            box = ax.boxplot(
                class_data,
                positions=[positions[pos_offset]],
                widths=0.6,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set_facecolor("lightblue")

            means = [np.mean(d) for d in [class_data]]
            for i, mean in enumerate(means):
                ax.text(
                    positions[pos_offset],
                    mean,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    color="red",
                    fontsize=12,
                )

    ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Set custom xticks for class labels
    ax.set_xticks(flattened_positions)
    ax.set_xticklabels(custom_class_labels * len(labels))

    # Add secondary x-axis for ensemble strategies
    strategy_positions = [
        np.mean(
            flattened_positions[
                i * len(custom_class_labels) : (i + 1) * len(custom_class_labels)
            ]
        )
        for i in range(len(labels))
    ]
    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(strategy_positions)
    secax.set_xticklabels(labels)
    secax.spines["bottom"].set_position(
        ("outward", 20)
    )  # Move the secondary x-axis outward
    secax.spines["bottom"].set_visible(False)  # Hide the secondary x-axis line
    secax.tick_params(
        axis="x", which="both", length=0
    )  # Hide the secondary x-axis ticks
    secax.set_xlabel("CDR per Strategy")

    ax.set_ylim(y_min, y_max)


# Plot EfficientNet models
plot_data(
    axes[0],
    efficientnet_data,
    efficientnet_labels,
    "EfficientNet-b2 Models",
    ylabel="F1 Score",
)

# Plot MobileViT models
plot_data(axes[1], mobilevit_data, mobilevit_labels, "MobileViT-s Models")

plt.tight_layout()
plt.savefig(
    f"plots/detailed_custom_comparative_f1_scores_boxplot.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()
plt.close()
