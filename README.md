# Wildfire Spread Prediction

Deep-learning project for next-day wildfire spread forecasting using multimodal geospatial data.

## Overview

This project investigates whether pretrained geospatial representations can improve next-day wildfire spread prediction.

Our baseline uses the **Next Day Wildfire Spread (NDWS)** dataset and an **Attention U-Net** segmentation model. We plan to compare a conventional multimodal model using the original NDWS environmental variables against a model augmented with **AlphaEarth satellite embeddings**.

The main prediction task is:

**current wildfire and environmental conditions → next-day wildfire mask**

## Current Method

The baseline model follows an Attention U-Net architecture.

The input contains spatially aligned wildfire and environmental variables from NDWS. Attention gates are added to the U-Net skip connections so that the decoder can emphasize spatial features that are more relevant to predicted wildfire spread.

The output is a single-channel pixel-wise probability map representing wildfire occurrence on the following day.

Because wildfire pixels are highly imbalanced relative to non-fire pixels, training uses a combination of Binary Cross Entropy and Dice loss.

## Data

### Next Day Wildfire Spread

NDWS contains wildfire events from the contiguous United States between 2012 and 2020.

Each sample represents a 64 km × 64 km region at 1 km spatial resolution and contains environmental variables together with wildfire masks at time \(t\) and \(t+1\).

The prediction target is the wildfire mask at \(t+1\).

### AlphaEarth Satellite Embeddings

AlphaEarth provides pretrained annual geospatial embeddings derived from multiple Earth-observation data sources.

Each 10 m pixel is represented by a 64-dimensional embedding vector. Our project investigates whether these learned representations provide useful additional landscape information for wildfire spread prediction.

Spatial and temporal alignment between AlphaEarth embeddings and NDWS samples is currently under development.

## Planned Experiments

The core comparison is:

| Model | NDWS Inputs | AlphaEarth Embeddings |
| --- | --- | --- |
| Baseline Attention U-Net | Yes | No |
| Embedding-Augmented Attention U-Net | Yes | Yes |

Both models will use the same prediction task, training procedure, data split, and evaluation metrics so that the contribution of the pretrained representation can be evaluated directly.

## Evaluation

The current implementation plan uses segmentation-oriented metrics, including Dice score and related overlap metrics.

Training uses a hybrid Binary Cross Entropy and Dice loss to address the strong class imbalance between wildfire and non-wildfire pixels.

## Repository Structure

The repository is currently under active development. The planned organization separates data processing, model architecture, training, and evaluation:

```text
.
├── data/
├── models/
├── scripts/
├── notebooks/
├── evaluation/
├── README.md
└── requirements.txt
```

## Reproducibility

Training will use controlled random seeds and documented model and dataset configurations so that experiments can be repeated under the same settings.

## Status

Work in progress.

Current priorities are:

Attention U-Net implementation, NDWS and AlphaEarth data ingestion, spatial alignment of the two data sources, PyTorch Dataset/DataLoader implementation, and reproducible baseline training.

## References

Key references include the Next Day Wildfire Spread dataset, the Attention U-Net wildfire spread methodology, and AlphaEarth Foundations.

Full citations and links will be added as the implementation is finalized.