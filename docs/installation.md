# Installation

Mantpy requires **Python 3.11 or newer**.

## Stable release

Install Mantpy from PyPI:

```bash
pip install mantpy
```

## Optional extras

Model training, segmentation, and SpatialData integration are opt-in:

| Extra | Adds | Install |
|-------|------|---------|
| `gnn` | Graph deep learning — GraphMAE embeddings, node classification, denoising and attribution (PyTorch + PyTorch Geometric + Captum) | `pip install "mantpy[gnn]"` |
| `patch` | Learned image-patch representations (PyTorch + Lightning) | `pip install "mantpy[patch]"` |
| `spatial` | SpatialData integration | `pip install "mantpy[spatial]"` |
| `segment` | Cellpose-based cell segmentation | `pip install "mantpy[segment]"` |

Extras can be combined, e.g. `pip install "mantpy[gnn,spatial]"`.

## Checking the installation

```python
import mantpy as mt

print(mt.__version__)
```
