import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# ── Compatibility shim: HfFolder ──────────────────────────────────────────────
# Gradio imports HfFolder from huggingface_hub in gradio/oauth.py.
# HF Spaces pre-installs huggingface_hub >= 0.30 which removed HfFolder.
# Injecting a minimal shim BEFORE importing gradio restores the attribute so
# the import chain succeeds regardless of which versions the platform chose.
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "HfFolder"):
    class _HfFolderShim:
        @staticmethod
        def get_token():
            return _hf_hub.utils._headers.get_token() if hasattr(_hf_hub, "utils") else None
        @staticmethod
        def save_token(token: str) -> None:
            pass
        @classmethod
        def delete_token(cls) -> None:
            pass
    _hf_hub.HfFolder = _HfFolderShim

# ── Compatibility shim: gradio_client boolean schema bug ──────────────────────
# gradio_client.utils._json_schema_to_python_type crashes when it encounters
# "additionalProperties": true/false (a boolean) in a JSON schema.  It tries
# to recurse into the boolean and calls get_type(True), which does
# `if "const" in schema:` — TypeError because `in` doesn't work on bools.
# This comes from Gradio's own FileData Pydantic model and affects ALL
# Gradio 4.x / 5.x versions.  We monkey-patch the two broken functions
# BEFORE importing gradio so the schema parser never crashes.
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type


def _patched_get_type(schema):
    """Handle the case where schema is a bool instead of a dict."""
    if isinstance(schema, bool):
        return "Any"
    return _orig_get_type(schema)


def _patched_json_schema_to_python_type(schema, defs=None):
    """Handle bool schemas before recursing."""
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils.get_type = _patched_get_type
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr
from PIL import Image

from inference import CLASSES, CONFIG, predict_image, predict_with_cam
from report import generate_report

SAMPLE_IMAGE_PATH = str(Path(__file__).parent / "static" / "sample_xray.jpg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _normalize_file_path(file_input) -> Optional[str]:
    """Normalize the many shapes Gradio can hand us for a file input."""
    if isinstance(file_input, str) and file_input:
        return file_input
    if isinstance(file_input, Path):
        return str(file_input)
    if isinstance(file_input, dict):
        path_val = file_input.get("path")
        if isinstance(path_val, str) and path_val:
            return path_val
        name_val = file_input.get("name")
        if isinstance(name_val, str) and name_val:
            return name_val
    # Some Gradio versions pass an object with a .name attribute.
    if hasattr(file_input, "name") and isinstance(file_input.name, str) and file_input.name:
        return file_input.name
    return None


def _validate_extension(file_input) -> str:
    file_path = _normalize_file_path(file_input)
    if not file_path:
        raise gr.Error("Upload an image first.")
    suffix = Path(file_path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".dcm"}:
        raise gr.Error("Please upload a .jpg, .jpeg, .png, or .dcm image.")
    return file_path


def _pil_to_b64(pil_img: Image.Image, max_size: int = 80) -> str:
    """Return a base64 data-URI for embedding in HTML img tags."""
    thumb = pil_img.copy()
    thumb.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Single-image results renderer
# ---------------------------------------------------------------------------

def _render_results_html(confidences: Dict[str, float], threshold: float) -> str:
    sorted_items = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    positive_rows: List[str] = []
    negative_rows: List[str] = []
    top_label = sorted_items[0][0] if sorted_items else None
    top_score = sorted_items[0][1] if sorted_items else 0.0
    use_argmax = CONFIG.get("prediction_mode", "multilabel") == "argmax"

    NO_FINDING_LABEL = "No Finding"
    other_positives = [
        label for label, score in confidences.items()
        if label != NO_FINDING_LABEL and score >= threshold
    ] if not use_argmax else []
    suppress_no_finding = len(other_positives) > 0

    for label, score in sorted_items:
        pct = score * 100
        is_suppressed_no_finding = (label == NO_FINDING_LABEL and suppress_no_finding)
        if use_argmax:
            is_positive = (label == top_label)
        else:
            is_positive = (score >= threshold) and not is_suppressed_no_finding
        bar_class = "bar-positive" if is_positive else "bar-negative"

        suppression_note = (
            "<div class='suppressed-note'><em>Suppressed — other pathologies detected</em></div>"
            if is_suppressed_no_finding else ""
        )

        row_html = f"""
        <div class='result-row'>
          <div class='row-top'>
            <span class='disease'>{label}</span>
            <span class='pct'>{pct:.2f}%</span>
          </div>
          <div class='bar-shell'>
            <div class='bar-fill {bar_class}' style='width: {max(1.0, pct):.2f}%;'></div>
          </div>
          {suppression_note}
        </div>
        """

        if is_positive:
            positive_rows.append(row_html)
        else:
            negative_rows.append(row_html)

    if not positive_rows:
        no_finding_score = confidences.get(NO_FINDING_LABEL, 0.0)
        if no_finding_score >= threshold:
            positive_block = "<p class='empty'>✓ No pathologies detected above threshold.</p>"
        else:
            positive_block = "<p class='empty'>No findings above current threshold.</p>"
    else:
        positive_block = "".join(positive_rows)
    negative_block = "".join(negative_rows)

    if use_argmax and top_label:
        mode_text = f"Top-1 prediction: <strong>{top_label}</strong> ({top_score * 100:.2f}%)"
    else:
        mode_text = f"Threshold mode: showing labels with confidence >= {threshold:.2f}"

    return f"""
    <div class='results-wrap'>
      <div class='card' style='margin-bottom: 10px; font-weight: 600;'>{mode_text}</div>
      <div class='section-head'>Positive Findings ({len(positive_rows)})</div>
      <div class='card positive-card'>
        {positive_block}
      </div>
      <details class='card negative-card' open>
        <summary>Negative / Below Threshold ({len(negative_rows)})</summary>
        <div class='neg-content'>{negative_block}</div>
      </details>
    </div>
    """


# ---------------------------------------------------------------------------
# Batch comparison renderer
# ---------------------------------------------------------------------------

def _render_batch_html(
    all_confidences: List[Dict[str, float]],
    filenames: List[str],
    thumbnails: List[Optional[Image.Image]],
    threshold: float,
) -> str:
    if not all_confidences:
        return "<div class='empty-th'>Run comparison to see results.</div>"

    NO_FINDING = "No Finding"

    # Build header row with thumbnails + filenames
    header_cells = "<th style='text-align:left; padding:6px 8px;'>Finding</th>"
    for fname, thumb in zip(filenames, thumbnails):
        short = fname[:20] + ("…" if len(fname) > 20 else "")
        img_tag = ""
        if thumb is not None:
            try:
                b64 = _pil_to_b64(thumb.convert("L"), max_size=80)
                img_tag = f"<img src='{b64}' style='display:block;margin:0 auto 4px;border-radius:6px;'/>"
            except Exception:
                pass
        header_cells += f"<th style='text-align:center;padding:6px 8px;min-width:90px;'>{img_tag}<span style='font-size:0.78rem'>{short}</span></th>"

    rows_html = ""
    for label in CLASSES:
        row = f"<td style='padding:5px 8px; font-weight:600; white-space:nowrap'>{label}</td>"
        for conf in all_confidences:
            score = conf.get(label, 0.0)
            pct = score * 100
            is_positive = score >= threshold and not (label == NO_FINDING)
            bg = "rgba(228,87,46,0.15)" if is_positive else "transparent"
            color = "#e4572e" if is_positive else "#6b7280"
            row += (
                f"<td style='text-align:center;padding:5px 8px;"
                f"background:{bg};color:{color};font-weight:700'>"
                f"{pct:.1f}%</td>"
            )
        rows_html += f"<tr>{row}</tr>"

    return f"""
    <div style='overflow-x:auto'>
      <table style='border-collapse:collapse;width:100%;font-size:0.85rem;font-family:IBM Plex Sans,sans-serif'>
        <thead>
          <tr style='background:#10243f;color:white'>{header_cells}</tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """


# ---------------------------------------------------------------------------
# Event handlers — single image tab
# ---------------------------------------------------------------------------

def _make_preview(file_input):
    file_path = _normalize_file_path(file_input)
    if not file_path:
        return None, "Path is empty"
    try:
        image = Image.open(file_path).convert("L")
    except Exception as e:
        return None, f"Image.open failed: {e}"
    image.thumbnail((300, 300))
    return image, None


def on_file_change(file_path):
    normalized_path = _normalize_file_path(file_path)
    if not normalized_path:
        return None, None, "", gr.update(interactive=False)

    try:
        validated_path = _validate_extension(normalized_path)
    except gr.Error as exc:
        raise gr.Error(str(exc))

    preview, err = _make_preview(validated_path)
    if preview is None:
        raise gr.Error(f"Could not read image preview. Please upload a valid image file. Details: {err}")

    size = os.path.getsize(validated_path)
    file_name = os.path.basename(validated_path)
    meta_md = (
        f"**File:** {file_name}  \n"
        f"**Size:** {_format_size(size)}  \n"
        "**Preview mode:** Grayscale"
    )
    return preview, preview, meta_md, gr.update(interactive=True)


def on_analyze(file_input, threshold: float):
    """Run inference + Grad-CAM. Returns (html, cache, cam_image, pdf_btn_update)."""
    file_path = _normalize_file_path(file_input)
    validated_path = _validate_extension(file_path)
    try:
        confidences, cam_image = predict_with_cam(validated_path)
    except FileNotFoundError as exc:
        raise gr.Error(
            "Model weights are not configured. Place best_model.pt in the project root "
            "or set MODEL_HF_REPO_ID / MODEL_GITHUB_URL before running app.py."
        ) from exc
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc

    html = _render_results_html(confidences, threshold)
    return html, confidences, cam_image, gr.update(visible=True)


def on_threshold_change(threshold: float, cached_confidences):
    if CONFIG.get("prediction_mode") == "argmax":
        return gr.update()
    if not cached_confidences:  # handles both None and {}
        return "<div class='empty-th'>Run analysis to see results.</div>"
    return _render_results_html(cached_confidences, threshold)


def on_sample_click():
    """Load the bundled sample X-ray."""
    if not Path(SAMPLE_IMAGE_PATH).exists():
        raise gr.Error("Sample image not found. Add static/sample_xray.jpg to the repo.")
    return on_file_change(SAMPLE_IMAGE_PATH)


def on_download_report(file_input, cached_confidences, threshold: float):
    """Generate PDF and return as a temporary file for gr.File download."""
    if not cached_confidences:  # handles both None and {}
        raise gr.Error("Run analysis first before downloading the report.")
    file_path = _normalize_file_path(file_input) or ""
    pdf_buf = generate_report(file_path, cached_confidences, threshold)

    # Write to a named temp file so Gradio can serve it.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="xray_report_")
    tmp.write(pdf_buf.read())
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Event handlers — batch tab
# ---------------------------------------------------------------------------

def on_batch_analyze(file_list, threshold: float):
    """Run predict_image on up to 4 images and return comparison HTML."""
    if not file_list:
        raise gr.Error("Upload at least one image.")
    paths = [_normalize_file_path(f) for f in file_list if _normalize_file_path(f)]
    if len(paths) > 4:
        raise gr.Error("Maximum 4 images for batch comparison.")

    all_confidences = []
    filenames = []
    thumbnails = []

    for path in paths:
        try:
            _validate_extension(path)
        except gr.Error as exc:
            raise gr.Error(f"File '{Path(path).name}': {exc}") from exc
        try:
            conf = predict_image(path)
        except Exception as exc:
            raise gr.Error(f"Inference failed for '{Path(path).name}': {exc}") from exc
        all_confidences.append(conf)
        filenames.append(Path(path).name)
        try:
            thumb = Image.open(path)
            thumbnails.append(thumb)
        except Exception:
            thumbnails.append(None)

    return _render_batch_html(all_confidences, filenames, thumbnails, threshold)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --bg-1: #f8fbff;
  --bg-2: #eef6ff;
  --ink: #10243f;
  --muted: #5f6b7a;
  --accent: #e4572e;
  --accent-soft: #f4a261;
  --cool: #2b6cb0;
  --panel: rgba(255, 255, 255, 0.88);
  --line: #d6e3f3;
}

.gradio-container {
  background:
    radial-gradient(circle at 10% 5%, rgba(43, 108, 176, 0.14), transparent 45%),
    radial-gradient(circle at 95% 15%, rgba(228, 87, 46, 0.12), transparent 38%),
    linear-gradient(160deg, var(--bg-1), var(--bg-2));
  font-family: 'IBM Plex Sans', sans-serif;
  color: var(--ink);
}

.banner {
  border: 2px solid #f6ad55;
  background: #fff4e5;
  color: #7b341e;
  border-radius: 14px;
  padding: 14px 16px;
  font-weight: 600;
}

.app-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.9rem;
  font-weight: 700;
  margin-bottom: 6px;
}

.badge {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 4px 10px;
  margin-right: 8px;
  font-size: 0.82rem;
  color: var(--cool);
  background: #ffffff;
}

#upload_zone {
  border: 2px dashed #8ea6c1;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.92);
}

.results-wrap .section-head {
  font-weight: 700;
  color: var(--ink);
  margin: 8px 0;
}

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px 12px;
}

.positive-card {
  border-left: 5px solid var(--accent);
}

.negative-card summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--muted);
}

.neg-content {
  margin-top: 8px;
}

.result-row {
  margin: 8px 0 12px;
}

.row-top {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 5px;
}

.disease {
  font-weight: 600;
  color: var(--ink);
}

.pct {
  font-weight: 700;
  color: #1f2937;
}

.bar-shell {
  height: 10px;
  border-radius: 999px;
  background: #e5edf7;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 250ms ease;
}

.bar-positive {
  background: linear-gradient(90deg, var(--accent-soft), var(--accent));
}

.bar-negative {
  background: #a0aec0;
}

.empty {
  color: var(--muted);
  margin: 4px 0;
}

.suppressed-note {
  font-size: 0.78rem;
  color: #9ca3af;
  margin-top: 2px;
  padding-left: 2px;
}

.footer {
  margin-top: 14px;
  font-size: 0.9rem;
  color: #45556a;
}

@media (max-width: 900px) {
  .app-title {
    font-size: 1.5rem;
  }
}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Chest X-Ray Multi-Label Classifier", css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <div class='banner'>
            This tool is for research/simulation only. Not a substitute for professional medical diagnosis.
        </div>
        """
    )
    gr.HTML(
        """
        <div class='app-title'>Chest X-Ray Multi-Label Pathology Classifier</div>
        <span class='badge'>Model: EfficientNet-B0</span>
        <span class='badge'>Dataset: NIH ChestX-ray14</span>
        <span class='badge'>Grad-CAM</span>
        <span class='badge'>DICOM</span>
        """
    )

    with gr.Tabs():
        # ── Tab 1: Single-image analysis ────────────────────────────────────
        with gr.Tab("Single Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_file = gr.File(
                        label="Drop your chest X-ray here (jpg / png / dcm)",
                        file_types=[".jpg", ".jpeg", ".png", ".dcm"],
                        type="filepath",
                        elem_id="upload_zone",
                    )
                    sample_btn = gr.Button("Try Sample X-Ray", variant="secondary")
                    meta_out = gr.Markdown()
                    analyze_btn = gr.Button("Analyze X-Ray", variant="primary", interactive=False)
                    report_btn = gr.Button("Download PDF Report", variant="secondary", visible=False)
                    report_file = gr.File(label="Report", visible=False)

                with gr.Column(scale=1):
                    preview_out = gr.Image(label="Grayscale Preview", type="pil")

            threshold = gr.Slider(
                minimum=CONFIG["threshold_min"],
                maximum=CONFIG["threshold_max"],
                value=CONFIG["default_threshold"],
                step=CONFIG["threshold_step"],
                label="Decision Threshold",
                interactive=CONFIG.get("prediction_mode", "multilabel") != "argmax",
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Image Preview")
                    result_preview = gr.Image(label="Analyzed Image", type="pil")

                with gr.Column(scale=1):
                    gr.Markdown("### Grad-CAM Heatmap")
                    cam_out = gr.Image(label="Grad-CAM (top class)", type="pil")

                with gr.Column(scale=2):
                    gr.Markdown("### Inference Results (Sorted by Confidence)")
                    results_html = gr.HTML("<div class='empty-th'>Run analysis to see results.</div>")

            cache_state = gr.State(None)  # {} breaks gradio_client JSON schema parser; None is safe

            image_file.change(
                on_file_change,
                inputs=[image_file],
                outputs=[preview_out, result_preview, meta_out, analyze_btn],
            )

            sample_btn.click(
                on_sample_click,
                inputs=[],
                outputs=[preview_out, result_preview, meta_out, analyze_btn],
            )

            analyze_btn.click(
                on_analyze,
                inputs=[image_file, threshold],
                outputs=[results_html, cache_state, cam_out, report_btn],
                show_progress="full",
            )

            threshold.change(
                on_threshold_change,
                inputs=[threshold, cache_state],
                outputs=[results_html],
            )

            report_btn.click(
                on_download_report,
                inputs=[image_file, cache_state, threshold],
                outputs=[report_file],
            )

            report_file.change(
                lambda f: gr.update(visible=f is not None),
                inputs=[report_file],
                outputs=[report_file],
            )

        # ── Tab 2: Batch comparison ──────────────────────────────────────────
        with gr.Tab("Batch Comparison"):
            gr.Markdown("Upload **2–4 chest X-rays** to compare side-by-side confidence scores.")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        label="Upload 2–4 X-Rays (jpg / png / dcm)",
                        file_count="multiple",
                        file_types=[".jpg", ".jpeg", ".png", ".dcm"],
                        type="filepath",
                    )
                    batch_threshold = gr.Slider(
                        minimum=CONFIG["threshold_min"],
                        maximum=CONFIG["threshold_max"],
                        value=CONFIG["default_threshold"],
                        step=CONFIG["threshold_step"],
                        label="Decision Threshold",
                    )
                    batch_btn = gr.Button("Compare All", variant="primary")

            batch_results = gr.HTML("<div class='empty-th'>Upload images and click Compare All.</div>")

            batch_btn.click(
                on_batch_analyze,
                inputs=[batch_files, batch_threshold],
                outputs=[batch_results],
                show_progress="full",
            )

    gr.HTML(
        """
        <div class='footer'>
            Model output: 20 sigmoid probabilities with multi-label threshold-based prediction.
            Grad-CAM heatmaps highlight regions driving the top prediction.
            <br>DICOM (.dcm) files are supported natively.
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
