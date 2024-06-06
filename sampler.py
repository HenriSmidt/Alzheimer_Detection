import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def create_generic_weighted_sampler(dataset, smoothing=1.0, strategy='sqrt'):
    """
    Creates a weighted sampler for any dataset.

    Parameters:
    - dataset: The dataset object, which must have a method get_labels()
    - smoothing: A smoothing factor to add to class counts to avoid harsh weights
    - strategy: The strategy for weight computation ('sqrt', 'log', 'exp', 'smoothed', 'custom')

    Returns:
    - sampler: A WeightedRandomSampler object
    """
    labels = dataset.get_labels()
    class_sample_count = np.array([len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)])

    if strategy == 'sqrt':
        weight = 1.0 / np.sqrt(class_sample_count)
    elif strategy == 'log':
        weight = 1.0 / np.log1p(class_sample_count)
    elif strategy == 'exp':
        weight = np.exp(-class_sample_count)
    elif strategy == 'smoothed':
        weight = 1.0 / (class_sample_count + smoothing)
    elif strategy == 'custom':
        weight = 1.0 / np.sqrt(class_sample_count) + 1.0 / np.log1p(class_sample_count)
    else:
        raise ValueError("Unsupported strategy. Choose from 'sqrt', 'log', 'exp', 'smoothed', 'custom'.")

    samples_weight = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler
