import os
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

from model import EfficientNetXRay


CONFIG = {
    "image_size": 224,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "default_threshold": 0.50,

    "threshold_min": 0.10,
    "threshold_max": 0.60,
    "threshold_step": 0.05,
    "num_classes": 20,
    "prediction_mode": "multilabel",
}

CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Pneumoperitoneum",
    "Pneumomediastinum",
    "Subcutaneous Emphysema",
    "Tortuous Aorta",
    "Calcification of the Aorta",
    "No Finding",
]


TRANSFORM = T.Compose(
    [
        T.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=CONFIG["imagenet_mean"], std=CONFIG["imagenet_std"]),
    ]
)


class ModelManager:
    def __init__(self) -> None:
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

    @property
    def device(self) -> torch.device:
        """Public accessor for the inference device."""
        return self._device

    @staticmethod
    def _download_from_github(url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, destination)
        return destination

    def _resolve_weights_path(self) -> Path:
        local_path = Path(os.getenv("MODEL_LOCAL_PATH", "best_model.pt"))
        if local_path.exists():
            return local_path

        hf_repo_id = os.getenv("MODEL_HF_REPO_ID", "").strip()
        hf_filename = os.getenv("MODEL_HF_FILENAME", "best_model.pt").strip()
        if hf_repo_id:
            downloaded = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename)
            return Path(downloaded)

        github_url = os.getenv("MODEL_GITHUB_URL", "").strip()
        if github_url:
            cache_path = Path(".cache") / "models" / "best_model.pt"
            if cache_path.exists():
                return cache_path
            return self._download_from_github(github_url, cache_path)

        raise FileNotFoundError(
            "No model weights found. Place best_model.pt in project root or set "
            "MODEL_HF_REPO_ID / MODEL_GITHUB_URL environment variables."
        )

    def load(self) -> EfficientNetXRay:
        with self._lock:
            if self._model is not None:
                return self._model

            model = EfficientNetXRay(num_classes=CONFIG["num_classes"], dropout=0.3)
            weights_path = self._resolve_weights_path()

            # weights_only=True requires PyTorch >= 1.13; fall back for older versions.
            try:
                state = torch.load(weights_path, map_location=self._device, weights_only=True)
            except TypeError:
                state = torch.load(weights_path, map_location=self._device)

            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            # Support checkpoints saved via DataParallel with 'module.' prefixes.
            cleaned_state = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state.items()
            }

            model.load_state_dict(cleaned_state, strict=True)
            model.eval()
            model.to(self._device)

            self._model = model
            return model


model_manager = ModelManager()


def _load_image(path: str) -> Image.Image:
    """
    Load an image from disk, with DICOM support.

    For .dcm files: reads via pydicom, applies VOI LUT, normalises to uint8,
    and returns a grayscale PIL image converted to RGB.
    For all other extensions: falls back to ``Image.open().convert("RGB")``.
    """
    if Path(path).suffix.lower() == ".dcm":
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError as exc:
            raise ImportError(
                "pydicom is required for DICOM support. "
                "Install it with: pip install pydicom"
            ) from exc

        ds = pydicom.dcmread(path)
        pixel_array = apply_voi_lut(ds.pixel_array.astype(float), ds)
        # Normalise to [0, 255] uint8.
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        pixel_array = pixel_array.astype("uint8")
        pil_img = Image.fromarray(pixel_array, mode="L").convert("RGB")
        return pil_img

    return Image.open(path).convert("RGB")


def predict_image(image_path: str) -> Dict[str, float]:
    model = model_manager.load()

    pil_image = _load_image(image_path)
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(model_manager.device)

    with torch.no_grad():
        output = model(tensor)
        probs = output[0].cpu().numpy().tolist()

    return {label: float(score) for label, score in zip(CLASSES, probs)}


def predict_with_cam(
    image_path: str,
    class_index: Optional[int] = None,
) -> Tuple[Dict[str, float], "PIL.Image.Image"]:
    """
    Run inference AND generate a Grad-CAM heatmap for the top class.

    Parameters
    ----------
    image_path:
        Path to the image file (jpg, png, or dcm).
    class_index:
        If None, uses the argmax of the sigmoid output (highest probability
        class) as the CAM target.  Pass an explicit index to explain a
        specific class.

    Returns
    -------
    (confidences dict, Grad-CAM PIL Image)
    """
    from gradcam import generate_cam  # local import to avoid circular deps at module load

    model = model_manager.load()
    pil_image = _load_image(image_path)
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(model_manager.device)

    # Forward pass for probabilities (no_grad for speed).
    with torch.no_grad():
        output = model(tensor)
        probs = output[0].cpu().numpy().tolist()

    confidences = {label: float(score) for label, score in zip(CLASSES, probs)}

    if class_index is None:
        class_index = int(max(range(len(probs)), key=lambda i: probs[i]))

    # Grad-CAM requires gradients; generate_cam handles enable_grad internally.
    cam_image = generate_cam(
        model=model,
        tensor_input=tensor,
        class_index=class_index,
        alpha=0.5,
        original_pil=pil_image,
    )

    return confidences, cam_image
