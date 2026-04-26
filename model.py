import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetXRay(nn.Module):
    """
    EfficientNet-B0 multi-label classifier for 20 chest X-ray pathology labels.

    WARNING: ``pretrained=False`` (the default for inference) means the EfficientNet
    backbone starts from *random* ImageNet weights — the backbone weights are overwritten
    entirely by the checkpoint loaded via ``model.load_state_dict()``.  Never use this
    model for inference without first loading a trained checkpoint; random weights will
    produce meaningless sigmoid probabilities.
    """

    def __init__(self, num_classes: int = 20, dropout: float = 0.3, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head — must match the saved checkpoint exactly.
        # Architecture: Linear→BN→ReLU→Dropout→Linear→ReLU→Dropout→Linear→Sigmoid
        # (indices 0-8, matching keys classifier.0, .1, .4, .7 in the state_dict)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),        # 0
            nn.BatchNorm1d(512),         # 1
            nn.ReLU(inplace=True),       # 2
            nn.Dropout(p=dropout),       # 3
            nn.Linear(512, 256),         # 4
            nn.ReLU(inplace=True),       # 5
            nn.Dropout(p=dropout),       # 6
            nn.Linear(256, num_classes), # 7
            nn.Sigmoid(),                # 8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
