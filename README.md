# Wildfire Spread Prediction

Deep-learning project for next-day wildfire spread forecasting using multimodal geospatial data and pretrained geospatial representations.

**Quick links:** [Solution](reports/a3/03_Solution.pdf) · [Attention U-Net](src/attention_unet/model.py) · [Loss](src/attention_unet/losses.py) · [Training](src/attention_unet/engine.py) · [Dataset](scripts/wsts_aef_dataset.py) · [Tests](tests/) · [Requirements](requirements.txt)

## Overview

This project investigates whether pretrained geospatial representations can improve next-day wildfire spread prediction.

Our baseline uses the **WildfireSpreadTS (WSTS)** dataset and an **Attention U-Net** segmentation model. We compare a conventional model using the preprocessed WSTS environmental and wildfire features against the same architecture augmented with **64-dimensional AlphaEarth Foundations (AEF) embeddings**.

The main prediction task remains:

**current wildfire and environmental conditions → next-day wildfire mask**

The current proposed method is documented in the [Milestone 3 solution](reports/a3/03_Solution.pdf).

## Current Method

The model follows a four-level **Attention U-Net** with double-convolution encoder/decoder blocks, a 1024-channel bridge, and additive attention gates on the skip connections.

After WSTS preprocessing, each sample contains **40 predictor channels**. The proposed model concatenates these with **64 spatially aligned AEF channels**, producing a 104-channel tensor:

```text
WSTS features   x : [B,  40, H, W]
AEF embeddings  e : [B,  64, H, W]
                         │
                         ▼
model input        [B, 104, H, W]
                         │
                         ▼
Attention U-Net    [B,   1, H, W]
```

The output is a single-channel pixel-wise probability map for next-day active fire.

Because wildfire pixels are highly imbalanced relative to non-fire pixels, training uses a hybrid **Binary Cross Entropy + Dice loss**. Segmentation evaluation currently supports IoU, Dice/F1, precision, and recall.

Implementation:

- [Attention U-Net](src/attention_unet/model.py)
- [BCE–Dice loss](src/attention_unet/losses.py)
- [Segmentation metrics](src/attention_unet/metrics.py)
- [Training/validation engine](src/attention_unet/engine.py)
- [Checkpointed fit loop](src/attention_unet/fit.py)
- [Reproducibility utilities](src/attention_unet/reproducibility.py)

## Data

### WildfireSpreadTS

WildfireSpreadTS contains georeferenced daily wildfire observations from the western United States between 2018 and 2021.

The source GeoTIFFs contain **23 raw channels** describing active fire, vegetation, weather, forecasts, topography, and land cover. The reference preprocessing expands categorical land cover and derives an active-fire-presence channel, yielding **40 model input channels**.

Our loader builds on the original WildfireSpreadTS PyTorch implementation:

- [WSTS + AEF Dataset](scripts/wsts_aef_dataset.py)
- [Vendored FireSpreadDataset](scripts/wsts_vendor/FireSpreadDataset.py)
- [HDF5 conversion](scripts/04_make_hdf5.py)

### AlphaEarth Foundations Embeddings

AlphaEarth provides annual 64-dimensional pretrained geospatial embeddings at approximately 10 m spatial resolution.

For each wildfire event, AEF embeddings are aligned to the exact 375 m WSTS grid. Fine-resolution embedding vectors within each WSTS pixel are reduced using a cosine medoid before being passed through the same crop/flip/rotation operations as the WSTS features.

Relevant pipeline code:

- [AEF medoid and grid utilities](scripts/aef_utils.py)
- [WSTS event indexing](scripts/01_build_event_index.py)
- [AEF export](scripts/02_export_aef_medoids.py)
- [Medoid validation](scripts/03_validate_aef_medoids.py)
- [Pixel-alignment smoke test](scripts/05_smoke_test_dataset.py)

## Planned Experiments

The core comparison is:

| Model | WSTS Inputs | AlphaEarth Embeddings | Input Channels |
| --- | :---: | :---: | ---: |
| Baseline Attention U-Net | ✓ | — | 40 |
| Embedding-Augmented Attention U-Net | ✓ | ✓ | 104 |

Both models use the same Attention U-Net architecture so that the effect of adding pretrained geospatial representations can be evaluated directly.

Final hyperparameters, benchmarking protocol, additional baselines, and compute configuration will be established during the evaluation milestone.

## Evaluation

The implementation supports segmentation-oriented metrics including:

- Intersection over Union (IoU)
- Dice / F1 score
- Precision
- Recall

Training uses the hybrid BCE–Dice objective and retains the checkpoint with the best validation IoU. Threshold selection, loss weighting, early-stopping patience, and other experimental hyperparameters remain configurable rather than being hard-coded into the training engine.

## Prerequisites and Installation

The project is implemented in Python with PyTorch. Python package dependencies are maintained in [`requirements.txt`](requirements.txt) so the environment can evolve without duplicating dependency information in this document.

From the repository root:

```bash
python -m pip install -r requirements.txt
```

GPU-specific PyTorch installations may require a platform-appropriate build for the target CUDA environment.

Verify the local data-processing environment with:

```bash
python scripts/00_check_setup.py
```

See [`scripts/00_check_setup.py`](scripts/00_check_setup.py) for the current setup checks.

## Quick Validation

Run the model-side test suite from the repository root:

```bash
python -m pytest -q
```

The tests cover model forward/backward propagation, loss and metrics, 40/104-channel batch construction, training and validation epochs, reproducibility utilities, and validation-IoU checkpointing.

Example model construction:

```python
from src.attention_unet.model import AttentionUNet

baseline = AttentionUNet(n_in_channels=40)
aef_model = AttentionUNet(n_in_channels=104)
```

The model-side training API is independent of dataset splitting and experiment-specific hyperparameters, allowing the same implementation to be reused as the evaluation protocol evolves.

## Repository Structure

The repository separates dataset preparation from the model-side implementation and project reports:

```text
.
├── configs/
│   └── experiment_configs_here
├── data/
│   └── sample_data_here
├── notebooks/
│   └── 01_eda.ipynb
├── reports/
│   ├── a1/
│   ├── a2/
│   └── a3/
│       ├── 03_Solution.pdf
│       ├── 03_Solution.tex
│       └── bibliography.bib
├── scripts/
│   ├── wsts_vendor/
│   │   ├── FireSpreadDataset.py
│   │   └── utils.py
│   ├── 00_check_setup.py
│   ├── 01_build_event_index.py
│   ├── 02_export_aef_medoids.py
│   ├── 03_validate_aef_medoids.py
│   ├── 04_make_hdf5.py
│   ├── 05_smoke_test_dataset.py
│   ├── aef_utils.py
│   └── wsts_aef_dataset.py
├── src/
│   └── attention_unet/
│       ├── __init__.py
│       ├── model.py
│       ├── losses.py
│       ├── metrics.py
│       ├── reproducibility.py
│       ├── engine.py
│       └── fit.py
├── tests/
│   ├── test_unet.py
│   ├── test_loss_metrics.py
│   ├── test_engine.py
│   └── test_fit.py
├── requirements.txt
└── README.md
```

## Reproducibility

Random-number generation is controlled across Python, NumPy, PyTorch, and CUDA through the [reproducibility utilities](src/attention_unet/reproducibility.py).

Model architecture, data loading, training orchestration, and experimental hyperparameters are kept separate so that future benchmark configurations can change without modifying the core implementation.

Dependency declarations are centralized in [`requirements.txt`](requirements.txt), while experiment-specific settings can be maintained independently as the evaluation protocol is finalized.

## Status

Milestone 3 implementation currently includes:

- four-level Attention U-Net
- BCE–Dice loss and segmentation metrics
- reproducibility controls
- WSTS/AEF batch fusion
- training and validation engine
- best-validation-IoU checkpointing
- synthetic model-side integration tests
- WSTS/AEF preprocessing and alignment pipeline

Next steps focus on full-dataset integration validation and Milestone 4 hyperparameter selection, baseline benchmarking, and evaluation.

## License

Vendored WildfireSpreadTS components originate from the [WildfireSpreadTS reference implementation](https://github.com/SebastianGer/WildfireSpreadTS) and remain subject to the upstream project's licensing terms.

A project-wide license has not yet been specified.

## References

Project reports contain the complete bibliography and methodological discussion:

- [Milestone 1 — Proposal](reports/a1/main.pdf)
- [Milestone 2 — Background & Related Work](reports/a2/2_Background.pdf)
- [Milestone 3 — Solution Design & Implementation](reports/a3/03_Solution.pdf)
- [WildfireSpreadTS reference implementation](https://github.com/SebastianGer/WildfireSpreadTS)