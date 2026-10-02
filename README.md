# UHM-FI

UHM-FI is a medical image representation learning framework that combines hierarchical anatomical modulation with feature interleaving. The public code contains the final model implementation, annotation utilities, data preparation scripts, pre-training configuration, and downstream classification and segmentation entry points.

## Installation

Use Python 3.10 or newer. The pinned CUDA environment is described in `environment.yml`; a lighter pip installation is available in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data preparation

Prepare local manifests with the supplied scripts and update the dataset paths in the YAML configurations before running an experiment.

## Downstream evaluation

The downstream entry point accepts a locally trained UHM-FI checkpoint through `--checkpoint`. For example, after preparing the RSNA manifest:

```bash
python downstream.py \
  --config configs/downstream_rsna_linear.yaml \
  --checkpoint /path/to/pretrained_checkpoint.pt \
  --device cuda \
  --execute
```

The same entry point supports the classification and segmentation configurations under `configs/`. Running without `--execute` validates the configuration without starting training.

## Pre-training

Set the local pre-training manifest path in `configs/pretrain.yaml`, then run:

```bash
python train.py --config configs/pretrain.yaml --device cuda --execute
```

The model package can also be imported directly:

```python
from uhm_fi import UHMFI, UHMFIConfig

config = UHMFIConfig.from_yaml("configs/uhm_fi_resnet50.yaml")
model = UHMFI(config)
```

The source code is provided for research use. Dataset access remains subject to the terms of the original data providers.
