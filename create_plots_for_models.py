import os
import pandas as pd
import numpy as np
import ast
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

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
    return f1_score(y_true, y_pred, average='weighted')

# Dictionary to store F1 scores
f1_scores = {}

# Iterate over the folders and CSV files
for category in os.listdir(root_dir):
    category_path = os.path.join(root_dir, category)
    if not os.path.isdir(category_path):
        continue

    for model_name in os.listdir(category_path):
        model_path = os.path.join(category_path, model_name)
        if not os.path.isdir(model_path):
            continue

        f1_scores[f"{category}_{model_name}"] = []

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
                slice_preds = np.array([softmax(np.array(ast.literal_eval(pred))) for pred in df[f"slice_{slice_number}"]])
                combined_predictions += slice_preds

            # Calculate mean predictions
            mean_predictions = combined_predictions / len(slice_numbers)

            # Get true labels and predicted labels
            y_true = df["labels"].values
            y_pred = np.argmax(mean_predictions, axis=1)

            # Calculate weighted F1 score
            f1 = calculate_weighted_f1(y_true, y_pred)
            f1_scores[f"{category}_{model_name}"].append(f1)

# Plot box plots
plt.figure(figsize=(14, 7))
plt.title("F1 Scores for Different Strategies and Models")
plt.xlabel("Model Strategy")
plt.ylabel("Weighted F1 Score")
plt.xticks(rotation=45)

data = [f1_scores[key] for key in f1_scores]
labels = [key for key in f1_scores]

plt.boxplot(data, labels=labels)
plt.tight_layout()
plt.savefig("f1_scores_boxplot.pdf", format="pdf", bbox_inches="tight")
plt.show()

        # plot_filename = f"plots/f1_score_boxplot_{strategy}_{model}.pdf"
        # plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
        # plt.close()
