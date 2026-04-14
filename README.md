# AquaSelect

Source code for "AquaSelect: Selective Prediction via Learned Score Fusion for Reliable Underwater Image Classification"

## Notebooks (run in order)

| Notebook | Description |
|----------|-------------|
| `01a` | ConvNeXt-Tiny training on AQUA20 (3 seeds) |
| `01b` | DeiT-Small training on AQUA20 (3 seeds) |
| `02a` | Selective prediction — frozen backbone |
| `02b` | Selective prediction — unfrozen backbone |
| `03` | Ablation study + Grad-CAM + t-SNE + confusion matrices |
| `04` | Conformal prediction (RAPS) analysis |
| `05a` | ConvNeXt-Tiny training on Sea Animals |
| `05b` | Selective prediction on Sea Animals |

## Requirements

- Python, PyTorch, timm, scikit-learn
- NVIDIA T4 GPU (all experiments run on Kaggle)

## Data

- [AQUA20](https://huggingface.co/datasets/taufiktrf/AQUA20)
- [Sea Animals](https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste)
