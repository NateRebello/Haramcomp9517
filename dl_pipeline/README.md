# Deep Learning pipeline (Abdoali)

ResNet-18 from-scratch training + shared utilities + evaluation notebook.

## Setup (Windows, CUDA GPU)

```powershell
cd dl_pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -m ipykernel install --user --name comp9517-dl --display-name "COMP9517-DL"
```

## Data path

Edit `src/config.py` → `DATA_ROOT` to point at your local iNat subset
(`train/`, `val/`, `test/` with 500 category folders).

## Notebooks

| Notebook | Owner / purpose |
|----------|-----------------|
| `01_environment_check.ipynb` | GPU / package check |
| `02_data_pipeline.ipynb` | DataLoaders + visual check |
| `03_train_from_scratch.ipynb` | Abdoali — ResNet-18 from scratch |
| `05_evaluation.ipynb` | Abdoali — spec metrics on test set |

**Nate:** write your own pretrained training + Grad-CAM on your machine. You can reuse `src/` helpers (`build_model(..., pretrained=True)`, `train_one_epoch`, etc.) or write your own — your call. Do not commit `.venv/`, datasets, or `*.pt` checkpoints.
