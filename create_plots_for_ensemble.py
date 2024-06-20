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

weighted_or_average_f1 = 'weighted'
if weighted_or_average_f1 == 'weighted':
    average = 'weighted'
    ensemble_column_name = 'test_f1_weighted'
    ylabel = 'Weighted F1 Score'
elif weighted_or_average_f1 == 'macro':
    average = 'macro'
    ensemble_column_name = 'test_f1_individual'
    ylabel = 'Average F1 Score'



# Dictionary to store F1 scores
f1_scores = {}
f1_scores_merged = {}
f1_scores_single_slice = {}
f1_scores_single_slice_merged = {}

filtered_file_path = 'filtered_ensemble_results.csv'
filtered_df = pd.read_csv(filtered_file_path)


# Add ensemble results to the data
efficientnet_ensemble_data = [filtered_df[(filtered_df['model_name'] == 'efficientnet-b2') & 
                                          (filtered_df['ensemble_variant'] == variant)][ensemble_column_name].values 
                              for variant in ['simple', 'medium', 'advanced']]
mobilevit_ensemble_data = [filtered_df[(filtered_df['model_name'] == 'MobileVit-s') & 
                                       (filtered_df['ensemble_variant'] == variant)][ensemble_column_name].values 
                           for variant in ['simple', 'medium', 'advanced']]


efficientnet_labels = ["Simple", "Medium", "Advanced"]
mobilevit_labels = ["Simple", "Medium", "Advanced"]

efficientnet_means = [np.mean(data) for data in efficientnet_ensemble_data]
mobilevit_means = [np.mean(data) for data in mobilevit_ensemble_data]

# Determine y-axis limits
all_f1_scores = efficientnet_ensemble_data + mobilevit_ensemble_data
y_min = min([min(scores) for scores in all_f1_scores if len(scores) > 0]) - 0.03
y_max = max([max(scores) for scores in all_f1_scores if len(scores) > 0]) + 0.03

# Create subplots for original classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)

# Plot EfficientNet models
axes[0].boxplot(efficientnet_ensemble_data)
axes[0].set_title("F1 Scores for Different EfficientNet-b2 Ensembles")
axes[0].set_xlabel("Ensemble Strategy")
axes[0].set_ylabel(ylabel)
axes[0].set_xticks(range(1, len(efficientnet_labels) + 1))
axes[0].set_xticklabels(efficientnet_labels)
axes[0].set_ylim(y_min, y_max)

for i, mean in enumerate(efficientnet_means):
    axes[0].text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', color='red', fontsize=12)

# Plot MobileViT models
axes[1].boxplot(mobilevit_ensemble_data)
axes[1].set_title("F1 Scores for Different MobileViT-s Ensembles")
axes[1].set_xlabel("Ensemble Strategy")
axes[1].set_xticks(range(1, len(mobilevit_labels) + 1))
axes[1].set_xticklabels(mobilevit_labels)
axes[1].set_ylim(y_min, y_max)

for i, mean in enumerate(mobilevit_means):
    axes[1].text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', color='red', fontsize=12)

plt.tight_layout()
plt.savefig(f"plots/comparative_f1_scores_{average}_of_ensembles_boxplot.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()


