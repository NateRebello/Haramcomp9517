# Deep Learning pipeline (COMP9517)

Shared ResNet-18 utilities + notebooks. Abdoali owns from-scratch training; Nate owns ImageNet-pretrained fine-tuning + Grad-CAM.

## Data

Expected layout (500 classes, 40/10/10 images):

```text
Haramcomp9517/subset/{train,val,test}/<category_id>/*.jpg
```

`src/config.py` defaults `DATA_ROOT` to that repo-local `subset/`. Override with env var `INAT_DATA_ROOT` (needed on Colab after mounting Drive).

Do **not** commit `subset/`, `.venv/`, or `*.pt` checkpoints.

## Local setup (Windows, Python 3.12 — CPU inspect / editing)

Nate’s machine has Intel Arc only (no NVIDIA CUDA). Use a **Python 3.12** venv for Cursor notebook editing and path smoke checks; run real training on Colab.

```powershell
cd dl_pipeline
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name comp9517-dl-nate --display-name "COMP9517-DL-Nate"
```

Select kernel **COMP9517-DL-Nate** in Cursor. Expect `torch.cuda.is_available() == False` locally.

## Colab setup (T4 — graded training / eval / Grad-CAM)

1. Upload `subset/` (~2.5 GB) to Google Drive once (keep `train/`, `val/`, `test/` intact).
2. In Colab: Runtime → GPU (T4).
3. Run the Colab bootstrap cells at the top of `notebooks/01_environment_check.ipynb` (clone/pull branch, mount Drive, set `INAT_DATA_ROOT`, install deps).
4. Confirm `torch.cuda.is_available()` is `True` before training.

```python
import os
os.environ["INAT_DATA_ROOT"] = "/content/drive/MyDrive/YOUR_PATH/subset"  # edit me
```

Colab already ships a CUDA build of PyTorch; still `pip install -r requirements.txt`.

## Notebooks

| Notebook | Owner / purpose |
|----------|-----------------|
| `01_environment_check.ipynb` | GPU / package / data-path check |
| `02_data_pipeline.ipynb` | DataLoaders + visual check |
| `03_train_from_scratch.ipynb` | Abdoali — ResNet-18 from scratch |
| `04_train_pretrained.ipynb` | Nate — ResNet-18 ImageNet fine-tune |
| `05_evaluation.ipynb` | Spec metrics on test set (scratch + pretrained) |
| `06_gradcam_analysis.ipynb` | Nate — Grad-CAM analysis |

**Nate Colab Drive path:** `/content/drive/MyDrive/comp9517/subset`  
Checkpoints mirrored to `/content/drive/MyDrive/comp9517/checkpoints/resnet18_pretrained/`.

Reuse `src/` helpers (`build_model(..., pretrained=True)`, `train_one_epoch`, metrics) rather than forking parallel training loops.
