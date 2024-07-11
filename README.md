# Alzheimer's Disease Detection in MRI Scans using Computer Vision

## Project Description

This project focuses on the detection of Alzheimer's Disease (AD) in MRI scans using advanced computer vision techniques and machine learning algorithms. Alzheimer's Disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early detection is crucial for managing and potentially slowing the progression of the disease. By leveraging the power of deep learning and computer vision, this project aims to assist medical professionals in diagnosing Alzheimer's Disease with greater accuracy and efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Alzheimer's Disease is characterized by the accumulation of amyloid plaques and neurofibrillary tangles in the brain, which can be detected through MRI scans. Traditional methods of diagnosis rely heavily on manual interpretation by medical professionals, which can be time-consuming and subjective. This project aims to develop an automated system for detecting Alzheimer's Disease from MRI scans using state-of-the-art computer vision and deep learning techniques.

## Dataset

The dataset used in this project is the OASIS-1 (Open Access Series of Imaging Studies) dataset, which includes MRI scans from a cross-sectional collection of 416 subjects aged 18 to 96. The dataset includes both healthy individuals and patients diagnosed with Alzheimer's Disease, providing a comprehensive set of images for training and evaluating the model.

- **Source**: [OASIS-1](https://www.oasis-brains.org/)

**Data Imbalance**: The dataset exhibits a significant class imbalance with the majority being non-demented. To address this, various sampling strategies and custom weights were employed to balance the training data.

**Correlation Analysis**: Dementia levels were correlated with age, education, and socioeconomic status, revealing moderate to slight correlations.

**Data Preparation**: MRI scans were divided into 176 horizontal slices, grouped into three-channel images to retain 3D structure. These groups were used to train individual models, with the top-performing slices selected for further analysis.


## Methodology
This section outlines the methodologies employed in this study, addressing the challenges posed by the dataset and detailing the methods used to achieve the research objectives.
### Concept Overview

Two model architectures, EfficientNet-B2 and MobileVit-S, were selected for their compact size and performance. Initial models established a baseline, followed by combining predictions from different slice group models to enhance classification accuracy. Improvements focused on handling data imbalance and refining training labels using self-distilled soft labels.

### Handling Data Imbalance

Various sampling strategies were tested to address data imbalance, including custom weights calculated using different formulas. A custom sampling method generated and evaluated random weights to find an optimal balance for model training.

### Self-Distillation

Self-distilled soft labels were used to refine training, creating smoother targets for the model and enhancing generalization. This involved transforming logits with a log-softmax function and optimizing parameters like learning rate and self-distillation alpha.

### Prediction Fusion

Predictions from ten models, each trained on a different top-performing slice group, were averaged to leverage comprehensive spatial information and improve classification robustness.

### Feature Fusion

Feature maps from the ten models were combined using different ensemble strategies, including simple concatenation, advanced feature fusion with fully connected layers, and attention feature fusion with multi-head attention mechanisms.

### Implementation Details

Patient-wise data splitting was performed to ensure the integrity of patient data, with 75% for training, 12.5% for validation, and 12.5% for testing. Experiments were conducted using an Apple M3 chip with 18 GB of RAM.


## Results

In this section, we present the results obtained from applying the methods described in the previous chapter. We evaluated all strategies using the EfficientNet-B2 and MobileViT-S models.

### Sampling Strategies Comparison

We identified the top ten performing slice groups from the initial slice group evaluation. These groups are centered around slices 65, 86, 56, 95, 62, 35, 59, 74, 80, and 134, as shown below:

![Best Performing Slice Groups](plots/f1_score_per_slice.pdf)
*Figure: The best performing slice groups are around the middle of the MRI scan.*

### Sampling Strategies Performance

The performance of various sampling strategies was measured using macro F1 scores for both models. The results are as follows:

| Sampling Strategy | EfficientNet-B2 | MobileViT-S |
|-------------------|-----------------|-------------|
| Custom            | 0.59            | 0.62        |
| Log               | 0.48            | 0.51        |
| Sqrt              | 0.47            | 0.45        |
| Inverse           | 0.44            | 0.52        |

### Sampling Strategy, Self Distillation, and Prediction Fusion

We adapted models using sampling and self-distillation techniques, evaluating them with averaged predictions from ten models to ensure reliability. The average F1 scores across different experiments are illustrated below:

![F1 Score Variability](plots/custom_comparative_f1_scores_macro_boxplot.pdf)
*Figure: The effect of different strategies on F1 score for CDR 0.5 and CDR 1.*

![Detailed F1 Score Distribution](plots/detailed_custom_comparative_f1_scores_boxplot.pdf)
*Figure: Detailed F1 scores for CDR 0.5 and CDR 1.*

For both models, the Mild Dementia class remained challenging, with significant performance improvements observed when aggregating predictions from multiple slices. Custom sampling weights notably increased the mean F1 score for the EfficientNet-based model by 0.15, while self-distillation decreased it.

### Feature Fusion Ensembles

We evaluated feature fusion ensembles for both models, with the best results obtained using the self-distilled model with a custom sampler. The macro F1 scores for different ensemble strategies are summarized below:

![Macro F1 Scores for Ensembles](plots/comparative_f1_scores_macro_of_ensembles_boxplot.pdf)
*Figure: Comparison of F1 scores for different feature fusion strategies.*

For EfficientNet-B2, the Attention model achieved the highest F1 score for Very Mild Dementia. For MobileViT-S, the Advanced model significantly improved the F1 score for Mild Dementia, as detailed below:

![Detailed Ensemble F1 Scores](plots/detailed_comparative_f1_scores_ensembles_boxplot.pdf)
*Figure: Detailed F1 scores for feature fusion ensembles.*

These results highlight the potential of ensemble strategies and custom sampling techniques in improving model performance for dementia classification tasks.


## Installation

...

## Usage

...

## Contributing

...

## License

...

## Acknowledgements

- Open Access Series of Imaging Studies (OASIS) for providing the dataset.
- The open-source community for developing tools and libraries that facilitate machine learning research.
