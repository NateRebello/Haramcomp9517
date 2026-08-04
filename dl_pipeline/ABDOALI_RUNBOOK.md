# Abdoali runbook — pretrained fine-tune + Grad-CAM (RTX 2050)

Nate’s Colab GPU quota is exhausted and his PC is CPU-only. Please run the
**ImageNet-pretrained** ResNet-18 path on your RTX 2050 and send the artefacts back.

Code arrives via PR: `nate/pretrained-gradcam` → `deepl_learning_metrics`.

Do **not** commit `subset/`, `.venv/`, or `*.pt` files.

---

## 1. Get the code

```powershell
cd <your Haramcomp9517 clone>
git fetch origin
git checkout deepl_learning_metrics
git pull origin deepl_learning_metrics
# After merging Nate's PR on GitHub:
git pull origin deepl_learning_metrics
```

Or merge the PR branch locally:

```powershell
git fetch origin
git checkout deepl_learning_metrics
git merge origin/nate/pretrained-gradcam
```

## 2. Environment (your existing CUDA venv)

```powershell
cd dl_pipeline
.\.venv\Scripts\Activate.ps1
# If needed, reinstall stack (includes grad-cam):
pip install -r requirements.txt
```

Kernel in Cursor/Jupyter: **COMP9517-DL** (your RTX 2050 kernel).

Confirm:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and `NVIDIA GeForce RTX 2050`.

## 3. Point at your local subset

`src/config.py` defaults to repo-root `subset/`, or reads `INAT_DATA_ROOT`.

If your data is still at the old path, either:

**Option A — env var (PowerShell, before launching Jupyter):**

```powershell
$env:INAT_DATA_ROOT = "C:\Users\Abdoali\Comp9517\Group_project\inat_subset\subset"
```

**Option B — edit** `dl_pipeline/src/config.py` default temporarily (do not commit a machine-specific path if you can avoid it).

### Probe (must pass before training)

```powershell
python -c "from pathlib import Path; import os; p=Path(os.environ.get('INAT_DATA_ROOT', r'..\subset'))/'train'/'1022'; print(p, 'jpgs', sum(1 for f in p.iterdir() if f.suffix.lower()=='.jpg'))"
```

Expect ~**40** jpgs. If `0`, the folder tree is empty stubs — fix the data path before notebooks.

## 4. Run notebooks (in order)

Skip Colab bootstrap cells on your PC (they no-op or are Drive-oriented). Use local `src.config` + `INAT_DATA_ROOT`.

| Order | Notebook | What success looks like |
|------:|----------|-------------------------|
| 1 | `notebooks/01_environment_check.ipynb` | `cuda.is_available() == True`, 500 classes |
| 2 | `notebooks/02_data_pipeline.ipynb` | One train batch loads, e.g. `(8, 3, 224, 224)` |
| 3 | `notebooks/04_train_pretrained.ipynb` | Smoke 1 epoch OK → full ~15 epochs |
| 4 | `notebooks/05_evaluation.ipynb` | `results/resnet18_pretrained_metrics.json` |
| 5 | `notebooks/06_gradcam_analysis.ipynb` | PNGs under `results/gradcam_pretrained/` |

### Notebook 04 settings (already defaults)

- `build_model(..., pretrained=True)`
- AdamW `lr=1e-4`, `BATCH_SIZE=16`, AMP on
- If OOM: set `TRAIN_BATCH = 8` and re-run from the config cell
- Checkpoints: `dl_pipeline/checkpoints/resnet18_pretrained/best.pt` (+ `last.pt`)
- History: `dl_pipeline/results/resnet18_pretrained_history.json`

Smoke first; only then run the full training cell. Leave the machine awake until finished.

## 5. Send back to Nate (Drive / zip — not git)

Zip and share:

```text
dl_pipeline/checkpoints/resnet18_pretrained/best.pt
dl_pipeline/checkpoints/resnet18_pretrained/last.pt   (optional)
dl_pipeline/results/resnet18_pretrained_history.json
dl_pipeline/results/resnet18_pretrained_curves.png    (if present)
dl_pipeline/results/resnet18_pretrained_metrics.json
dl_pipeline/results/gradcam_pretrained/               (all PNGs + gradcam_analysis_notes.json)
```

If you fill in the Grad-CAM TODO findings in `gradcam_analysis_notes.json`, even better; otherwise Nate will interpret from the PNGs.

## 6. Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `Found no valid file for the classes ...` | Empty class folders — fix `INAT_DATA_ROOT` / re-copy subset with real `.jpg`s |
| CUDA OOM | `TRAIN_BATCH = 8`, keep AMP |
| `ModuleNotFoundError: gradcam` / `pytorch_grad_cam` | `pip install grad-cam` inside the venv |
| Scratch missing in 05 | OK — eval will skip scratch and still run pretrained |

Thanks — this unblocks the report comparison (scratch vs pretrained) and the Grad-CAM advanced direction.
