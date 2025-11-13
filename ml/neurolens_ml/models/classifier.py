"""EfficientNet classifier (timm) and a wrapper that also returns per-class CAMs."""
from pathlib import Path

import timm
import torch

from ..config import CLASSES


def create_classifier(name: str, pretrained: bool) -> torch.nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))


def load_classifier(name: str, checkpoint: str | Path) -> torch.nn.Module:
    model = create_classifier(name, pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


class ClassifierWithCam(torch.nn.Module):
    """Returns logits and one CAM per class.

    With global average pooling + one linear layer, Grad-CAM equals CAM = classifier weights x last feature maps,
    so the heatmap needs no backward pass and can be exported to ONNX.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.forward_features(x)                                           # [B, C, h, w]
        logits = self.model.forward_head(feats)                                          # [B, classes]
        cams = torch.einsum("kc,bchw->bkhw", self.model.get_classifier().weight, feats)  # [B, classes, h, w]
        return logits, cams
