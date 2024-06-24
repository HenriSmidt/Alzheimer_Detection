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

# weighted_or_average_f1 = 'weighted'
# if weighted_or_average_f1 == 'weighted':
#     average = 'weighted'
#     ensemble_column_name = 'test_f1_weighted'
#     ylabel = 'Weighted F1 Score'
# elif weighted_or_average_f1 == 'macro':
#     average = 'macro'
#     ensemble_column_name = 'test_f1_individual'
#     ylabel = 'Average F1 Score'



# # Dictionary to store F1 scores
# f1_scores = {}
# f1_scores_merged = {}
# f1_scores_single_slice = {}
# f1_scores_single_slice_merged = {}

# filtered_file_path = 'csvs/filtered_ensemble_results_custom_weights.csv'
# filtered_df = pd.read_csv(filtered_file_path)


# # Add ensemble results to the data
# efficientnet_ensemble_data = [filtered_df[(filtered_df['model_name'] == 'efficientnet-b2') & 
#                                           (filtered_df['ensemble_variant'] == variant)][ensemble_column_name].values 
#                               for variant in ['simple', 'medium', 'advanced']]
# mobilevit_ensemble_data = [filtered_df[(filtered_df['model_name'] == 'MobileVit-s') & 
#                                        (filtered_df['ensemble_variant'] == variant)][ensemble_column_name].values 
#                            for variant in ['simple', 'medium', 'advanced']]


# efficientnet_labels = ["Simple", "Advanced", "Attention"]
# mobilevit_labels = ["Simple", "Advanced", "Attention"]

# efficientnet_means = [np.mean(data) for data in efficientnet_ensemble_data]
# mobilevit_means = [np.mean(data) for data in mobilevit_ensemble_data]

# # Determine y-axis limits
# all_f1_scores = efficientnet_ensemble_data + mobilevit_ensemble_data
# y_min = min([min(scores) for scores in all_f1_scores if len(scores) > 0]) - 0.03
# y_max = max([max(scores) for scores in all_f1_scores if len(scores) > 0]) + 0.03

# # Create subplots for original classes
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5), sharey=True)

# # Plot EfficientNet models
# axes[0].boxplot(efficientnet_ensemble_data)
# axes[0].set_title("F1 Scores for Different EfficientNet-b2 Ensembles")
# axes[0].set_xlabel("Ensemble Strategy")
# axes[0].set_ylabel(ylabel)
# axes[0].set_xticks(range(1, len(efficientnet_labels) + 1))
# axes[0].set_xticklabels(efficientnet_labels)
# axes[0].set_ylim(y_min, y_max)

# for i, mean in enumerate(efficientnet_means):
#     axes[0].text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', color='red', fontsize=12)

# # Plot MobileViT models
# axes[1].boxplot(mobilevit_ensemble_data)
# axes[1].set_title("F1 Scores for Different MobileViT-s Ensembles")
# axes[1].set_xlabel("Ensemble Strategy")
# axes[1].set_xticks(range(1, len(mobilevit_labels) + 1))
# axes[1].set_xticklabels(mobilevit_labels)
# axes[1].set_ylim(y_min, y_max)

# for i, mean in enumerate(mobilevit_means):
#     axes[1].text(i + 1, mean, f'{mean:.2f}', ha='center', va='bottom', color='red', fontsize=12)

# plt.tight_layout()
# plt.savefig(f"plots/comparative_f1_scores_{average}_of_ensembles_boxplot.pdf", format="pdf", bbox_inches="tight")
# plt.show()
# plt.close()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dictionary to store F1 scores
f1_scores = {}
f1_scores_merged = {}
f1_scores_single_slice = {}
f1_scores_single_slice_merged = {}

filtered_file_path = 'csvs/filtered_ensemble_results_custom_weights.csv'
filtered_df = pd.read_csv(filtered_file_path)

# List of column names representing different classes
column_names = ['classification_report.0.f1-score', 'classification_report.1.f1-score', 'classification_report.2.f1-score']  # Replace with your actual class column names
class_labels = ['No AD', 'vm AD', 'm AD'] 

# Define ensemble variants
ensemble_variants = ['simple', 'medium', 'advanced']

# Prepare data
def prepare_ensemble_data(model_name):
    return {
        column_name: [filtered_df[(filtered_df['model_name'] == model_name) & 
                                  (filtered_df['ensemble_variant'] == variant)][column_name].values 
                      for variant in ensemble_variants]
        for column_name in column_names
    }

efficientnet_data = prepare_ensemble_data('efficientnet-b2')
mobilevit_data = prepare_ensemble_data('MobileVit-s')

# Plot settings
def plot_ensemble_data(ax, data, class_labels, title):
    all_f1_scores = [scores for class_scores in data.values() for scores in class_scores]
    y_min = min([min(scores) for scores in all_f1_scores if len(scores) > 0]) - 0.03
    y_max = max([max(scores) for scores in all_f1_scores if len(scores) > 0]) + 0.03

    # Create positions for each class within each strategy
    positions = []
    for i in range(len(ensemble_variants)):
        positions.extend(range(i * len(class_labels) + 1, (i + 1) * len(class_labels) + 1))
    
    for idx, (class_name, class_data) in enumerate(data.items()):
        pos_offset = idx
        class_positions = [positions[i * len(class_labels) + pos_offset] for i in range(len(ensemble_variants))]
        box = ax.boxplot(class_data, positions=class_positions, widths=0.6, patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('lightblue')

        means = [np.mean(d) for d in class_data]
        for i, mean in enumerate(means):
            ax.text(class_positions[i], mean, f'{mean:.2f}', ha='center', va='bottom', color='red', fontsize=12)
    
    ax.set_title(title)
    # ax.set_xlabel("Classes per Ensemble Strategy")
    ax.set_ylabel("F1 Score")
    
    # Set custom xticks for class labels
    ax.set_xticks(positions)
    ax.set_xticklabels(class_labels * len(ensemble_variants))
    
    # Add secondary x-axis for ensemble strategies
    strategy_positions = [np.mean(positions[i * len(class_labels):(i + 1) * len(class_labels)]) for i in range(len(ensemble_variants))]
    secax = ax.secondary_xaxis('bottom')
    secax.set_xticks(strategy_positions)
    secax.set_xticklabels(ensemble_variants)
    secax.spines['bottom'].set_position(('outward', 20))  # Move the secondary x-axis outward
    secax.set_xlabel("Classes per Ensemble Strategy")

    # Hide the secondary x-axis line and ticks
    secax.spines['bottom'].set_visible(False)
    secax.tick_params(axis='x', which='both', length=0)

    ax.set_ylim(y_min, y_max)

# Create subplots for original classes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

# Plot EfficientNet models
plot_ensemble_data(axes[0], efficientnet_data, class_labels, "EfficientNet-b2 Ensembles")

# Plot MobileViT models
plot_ensemble_data(axes[1], mobilevit_data, class_labels, "MobileViT-s Ensembles")

plt.tight_layout()
plt.savefig(f"plots/detailed_comparative_f1_scores_ensembles_boxplot.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()