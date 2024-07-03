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

## Model Architecture

...

## Results

...

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
