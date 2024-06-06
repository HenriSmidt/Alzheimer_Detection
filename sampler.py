import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

def create_generic_weighted_sampler(dataset, smoothing=0.0, strategy='inverse'):
    """
    Creates a weighted sampler for any dataset.

    Parameters:
    - dataset: The dataset object, which must have a method called 'get_labels'.
               The 'get_labels' method should return a list or array of labels corresponding to each sample in the dataset.
    - smoothing: A smoothing factor to add to class counts to avoid harsh weights (default: 0.0)
    - strategy: The strategy for weight computation ('inverse', 'sqrt', 'log', 'exp', 'custom') (default: 'inverse')

    Returns:
    - sampler: A WeightedRandomSampler object

    Raises:
    - AttributeError: If the dataset does not have a 'get_labels' method.
    - ValueError: If an unsupported strategy is provided.
    """
    # Check if the dataset has the get_labels method
    if not hasattr(dataset, 'get_labels') or not callable(getattr(dataset, 'get_labels')):
        raise AttributeError("The dataset object must have a method called 'get_labels' which returns a list or array of labels for each sample in the dataset.")

    labels = dataset.get_labels()
    class_sample_count = np.array([len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)])

    if strategy == 'inverse':
        weight = 1.0 / (class_sample_count + smoothing)
    elif strategy == 'sqrt':
        weight = 1.0 / np.sqrt(class_sample_count + smoothing)
    elif strategy == 'log':
        weight = 1.0 / np.log1p(class_sample_count + smoothing)
    elif strategy == 'exp':
        weight = np.exp(-(class_sample_count + smoothing))
    elif strategy == 'custom':
        weight = 1.0 / np.sqrt(class_sample_count + smoothing) + 1.0 / np.log1p(class_sample_count + smoothing)
    else:
        raise ValueError("Unsupported strategy. Choose from 'inverse', 'sqrt', 'log', 'exp', 'custom'.")

    samples_weight = np.array([weight[label] for label in labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    return sampler
