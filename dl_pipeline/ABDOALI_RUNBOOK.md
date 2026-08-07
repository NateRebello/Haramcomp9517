# Abdoali runbook — pretrained fine-tune + Grad-CAM (RTX 2050)

Canonical checklist for Abdoali’s Windows laptop (RTX 2050). Skip Colab entirely.

Do **not** commit `subset/`, `.venv/`, or `*.pt` files. Zip artefacts and send to Nate (Drive/WhatsApp), not via git.

---

## Assumptions (locked)

- Windows laptop; project already cloned (or cloneable)
- Subset at `C:\Users\Abdoali\Comp9517\Group_project\inat_subset\subset` with real `.jpg` files
- CUDA venv for kernel **COMP9517-DL (RTX2050)** at:
  `C:\Users\Abdoali\Comp9517\Group_project\dl_pipeline\.venv`
  (outside the git clone; not under `Haramcomp9517\dl_pipeline\.venv`)
- Do **not** use `COMP9517-DL-Nate` (CPU)

---

## A. Open terminal and get code

Open PowerShell. Go to the repo (adjust if clone path differs):

```powershell
cd C:\Users\Abdoali\Comp9517\Group_project\Haramcomp9517
```

If that folder does not exist, clone once:

```powershell
cd C:\Users\Abdoali\Comp9517\Group_project
git clone https://github.com/NateRebello/Haramcomp9517.git
cd Haramcomp9517
```

Pull Nate’s work onto the DL branch:

```powershell
git fetch origin
git checkout deepl_learning_metrics
git pull origin deepl_learning_metrics
git merge origin/nate/pretrained-gradcam
```

If merge asks for a message, save and close. If conflicts appear, stop and ask Nate.

Confirm notebooks exist:

```powershell
dir dl_pipeline\notebooks\04_train_pretrained.ipynb
dir dl_pipeline\ABDOALI_RUNBOOK.md
```

---

## B. Activate GPU environment

```powershell
cd C:\Users\Abdoali\Comp9517\Group_project\Haramcomp9517\dl_pipeline
& C:\Users\Abdoali\Comp9517\Group_project\dl_pipeline\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Must print: `True NVIDIA GeForce RTX 2050` (or similar NVIDIA name).

If `False`, stop — wrong Python/venv (do not use Nate’s CPU kernel).

---

## C. Lock the data path (same PowerShell window)

```powershell
$env:INAT_DATA_ROOT = "C:\Users\Abdoali\Comp9517\Group_project\inat_subset\subset"
python -c "from pathlib import Path; import os; p=Path(os.environ['INAT_DATA_ROOT'])/'train'/'1022'; print(p); print('jpgs', sum(1 for f in p.iterdir() if f.suffix.lower()=='.jpg'))"
```

Must show ~40 jpgs. If 0 or path missing, fix the subset path before any notebook.

Keep this PowerShell window open if you launch Jupyter from it so the env var sticks. In Cursor, set the var in the first code cell of each notebook **before** `import src.config`:

```python
import os
os.environ["INAT_DATA_ROOT"] = r"C:\Users\Abdoali\Comp9517\Group_project\inat_subset\subset"
```

---

## D. Open Cursor and pick the right kernel

Open the `Haramcomp9517` folder in Cursor.

Open notebooks from `dl_pipeline\notebooks\`.

Kernel for every notebook: **COMP9517-DL** (Abdoali’s CUDA venv / display name **COMP9517-DL (RTX2050)**).

Not `COMP9517-DL-Nate`.

---

## E. Run notebooks in this exact order

For each notebook: set `INAT_DATA_ROOT` cell first (if using Cursor), then Run All. Skip/ignore Colab bootstrap failures — on this PC those cells should print `IN_COLAB = False` and continue.

### E1 — `01_environment_check.ipynb`

Pass if: CUDA True, `DATA_ROOT` exists, train/val/test each 500 classes.

### E2 — `02_data_pipeline.ipynb`

Pass if: batch shape like `(8, 3, 224, 224)` and “Data pipeline OK”.

### E3 — `04_train_pretrained.ipynb` (the long one)

1. Run setup + config + data/model cells.
2. Run **§4 One-epoch smoke test** only first.
3. Pass if: finite loss, no CUDA OOM, finishes.
4. If OOM: set `TRAIN_BATCH = 8` in the config cell, re-run from config → data → smoke.
5. Then run **§5 Full fine-tuning** (~15 epochs). Do not close laptop; plug in power.

Pass if these files exist:

- `dl_pipeline\checkpoints\resnet18_pretrained\best.pt`
- `dl_pipeline\results\resnet18_pretrained_history.json`

### E4 — `05_evaluation.ipynb`

Pass if: `dl_pipeline\results\resnet18_pretrained_metrics.json` is written.

Scratch may be skipped if Abdoali’s from-scratch `best.pt` is missing — that is OK.

### E5 — `06_gradcam_analysis.ipynb`

Pass if: PNGs under `dl_pipeline\results\gradcam_pretrained\`.

Optionally edit TODO strings in `gradcam_analysis_notes.json`.

---

## F. Package results (do not git-commit `.pt`)

In PowerShell from `dl_pipeline`:

```powershell
mkdir ..\nate_pretrained_results -Force
Copy-Item checkpoints\resnet18_pretrained\best.pt ..\nate_pretrained_results\
Copy-Item results\resnet18_pretrained_history.json ..\nate_pretrained_results\ -ErrorAction SilentlyContinue
Copy-Item results\resnet18_pretrained_metrics.json ..\nate_pretrained_results\ -ErrorAction SilentlyContinue
Copy-Item results\resnet18_pretrained_curves.png ..\nate_pretrained_results\ -ErrorAction SilentlyContinue
Copy-Item -Recurse results\gradcam_pretrained ..\nate_pretrained_results\
Compress-Archive -Path ..\nate_pretrained_results\* -DestinationPath ..\nate_pretrained_results.zip -Force
```

Send `nate_pretrained_results.zip` to Nate (Drive/WhatsApp), **not** via git.

---

## Quick troubleshooting

| Symptom | Fix |
|---------|-----|
| `Found no valid file for the classes ...` | Empty class folders — fix `INAT_DATA_ROOT` / re-copy subset with real `.jpg`s |
| CUDA OOM | `TRAIN_BATCH = 8`, keep AMP |
| `ModuleNotFoundError: gradcam` / `pytorch_grad_cam` | `pip install grad-cam` inside the CUDA venv |
| Scratch missing in 05 | OK — eval will skip scratch and still run pretrained |
| Wrong kernel / `cuda.is_available() False` | Switch to **COMP9517-DL (RTX2050)**; activate `Group_project\dl_pipeline\.venv` |
