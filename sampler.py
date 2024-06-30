import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def create_generic_weighted_sampler(
    dataset, smoothing=0.0, strategy="inverse", custom_weights=None
):
    """
    Creates a weighted sampler for any dataset.

    Parameters:
    - dataset: The dataset object, which must have a method called 'get_labels'.
               The 'get_labels' method should return a list or array of labels corresponding to each sample in the dataset.
    - smoothing: A smoothing factor to add to class counts to avoid harsh weights (default: 0.0)
    - strategy: The strategy for weight computation ('inverse', 'sqrt', 'log', 'exp', 'custom') (default: 'inverse')
    - custom_weights: A dictionary of custom weights for each class (default: None)

    Returns:
    - sampler: A WeightedRandomSampler object

    Raises:
    - AttributeError: If the dataset does not have a 'get_labels' method.
    - ValueError: If an unsupported strategy is provided or if custom_weights are not valid.
    """
    # Check if the dataset has the get_labels method
    if not hasattr(dataset, "get_labels") or not callable(
        getattr(dataset, "get_labels")
    ):
        raise AttributeError(
            "The dataset object must have a method called 'get_labels' which returns a list or array of labels for each sample in the dataset."
        )

    labels = dataset.get_labels()
    unique_labels = np.unique(labels)
    class_sample_count = np.array(
        [len(np.where(np.array(labels) == t)[0]) for t in unique_labels]
    )

    if strategy == "inverse":
        weight = 1.0 / (class_sample_count + smoothing)
    elif strategy == "sqrt":
        weight = 1.0 / np.sqrt(class_sample_count + smoothing)
    elif strategy == "log":
        weight = 1.0 / np.log1p(class_sample_count + smoothing)
    elif strategy == "exp":
        weight = np.exp(-(class_sample_count + smoothing))
    elif strategy == "custom":
        if custom_weights is None:
            raise ValueError(
                "custom_weights must be provided when strategy is 'custom'."
            )
        if not isinstance(custom_weights, dict):
            raise ValueError("custom_weights must be a dictionary.")
        if len(custom_weights) != len(unique_labels):
            raise ValueError(
                "Number of custom weights must match the number of unique classes."
            )

        weight = np.array([custom_weights[label] for label in unique_labels])
    else:
        raise ValueError(
            "Unsupported strategy. Choose from 'inverse', 'sqrt', 'log', 'exp', 'custom'."
        )

    samples_weight = np.array(
        [weight[np.where(unique_labels == label)[0][0]] for label in labels]
    )
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler
