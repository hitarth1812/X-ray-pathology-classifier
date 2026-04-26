---
title: Chest X-ray Classifier
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.7.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Chest X-Ray Multi-Label Pathology Classifier (Simulation)

Single-page Gradio web app for multi-label chest X-ray inference with EfficientNet-B0 + sigmoid probabilities.

## Features

- Drag-and-drop or browse upload for `.jpg`, `.jpeg`, `.png`
- Grayscale image preview and file metadata (name + size)
- EfficientNet-B0 inference pipeline (`224x224`, ImageNet normalization)
- 20 pathology confidence scores with sorted horizontal bars
- Interactive threshold slider (`0.10` to `0.60`, step `0.05`)
- Positive findings highlighted; below-threshold findings greyed/collapsible
- Research disclaimer banner for safe usage context

## Class Labels

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural_Thickening
13. Pneumonia
14. Pneumothorax
15. Pneumoperitoneum
16. Pneumomediastinum
17. Subcutaneous Emphysema
18. Tortuous Aorta
19. Calcification of the Aorta
20. No Finding

## Project Structure

```
chest-xray-classifier/
├── app.py
├── model.py
├── inference.py
├── requirements.txt
├── README.md
└── static/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model Weights Resolution

The app resolves weights in this order:

1. Local file path `best_model.pt` (or `MODEL_LOCAL_PATH`)
2. Hugging Face Hub (`MODEL_HF_REPO_ID` + optional `MODEL_HF_FILENAME`)
3. GitHub Release direct URL (`MODEL_GITHUB_URL`)

### Example (Hugging Face Hub)

```powershell
$env:MODEL_HF_REPO_ID="your-org/your-model-repo"
$env:MODEL_HF_FILENAME="best_model.pt"
python app.py
```

### Example (GitHub Release)

```powershell
$env:MODEL_GITHUB_URL="https://github.com/<owner>/<repo>/releases/download/<tag>/best_model.pt"
python app.py
```

## Run Locally

```bash
python app.py
```

Open the local Gradio URL shown in terminal (typically `http://127.0.0.1:7860`).

## Hugging Face Spaces Deployment

> **Important — Model Weights:** `best_model.pt` is listed in `.gitignore` and will **not** be pushed to GitHub or Hugging Face Spaces automatically.
> You must make the weights available to the Space via one of:
> - Uploading the file directly inside the Space's file editor, **or**
> - Setting `MODEL_HF_REPO_ID` (+ optional `MODEL_HF_FILENAME`) as a Space Secret, **or**
> - Setting `MODEL_GITHUB_URL` as a Space Secret pointing to a public GitHub Release asset.
>
> Without one of these, the app will start but inference will fail with a `FileNotFoundError`.

1. Create a new Gradio Space.
2. Upload these files (excluding `best_model.pt` — handle it via env var as above).
3. Add model environment variable(s) in Space **Settings → Repository secrets**.
4. Verify the Space build log shows the model loaded successfully.

## Notes

- This is a simulation/research demo.
- Not for clinical diagnosis or medical decision making.
