import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from cycler import cycler
import random
import lightning.pytorch as pl
import numpy as np


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_plot_style():
    # Extract the first four colors from the Paired colormap
    paired_colors = plt.cm.Paired(range(10))
    selected_colors = np.concatenate(
        [paired_colors[:4], paired_colors[6:7], paired_colors[7:8]]
    )

    # Set the color cycle with the selected colors
    plt.rc("axes", prop_cycle=(cycler("color", selected_colors)))
    # Step 1: Register custom font with Matplotlib
    font_path = "/Users/henrismidt/Documents/Informatik/Fonts/libertinus/LibertinusSerif-Regular.otf"
    font_manager.fontManager.addfont(font_path)  # Register the font with Matplotlib

    # Step 2: Update Matplotlib's RC settings to use font by default
    plt.rcParams["font.family"] = "Libertinus Serif"

    # Set font sizes (common professional sizes + 2)
    plt.rcParams.update(
        {
            "axes.titlesize": 16,  # Title font size (14 + 2)
            "axes.labelsize": 13,  # Axis labels font size (12 + 2)
            "xtick.labelsize": 12,  # X-axis tick labels font size (10 + 2)
            "ytick.labelsize": 12,  # Y-axis tick labels font size (10 + 2)
            "legend.fontsize": 12,  # Legend font size (10 + 2)
            "figure.titlesize": 18,  # Figure title font size (16 + 2)
            "legend.title_fontsize": 13,  # Legend title font size (12 + 2)
            "axes.edgecolor": "lightgray",  # Axis edge color
            "axes.grid": False,  # Disable grid by default
            "xtick.top": False,  # Turn off top x-ticks
            "xtick.bottom": False,  # Turn off bottom x-ticks
            "ytick.left": False,  # Turn off left y-ticks
            "ytick.right": False,  # Turn off right y-ticks
        }
    )
    return selected_colors


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
