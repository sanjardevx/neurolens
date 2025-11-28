"""3D tumor segmenter: MONAI SegResNet, 4 MRI types in, 3 overlapping region masks out (TC, WT, ET)."""
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet


def create_segmenter(cfg: dict) -> SegResNet:
    # The BraTS 2018 winning layout (Myronenko), ~4.7M params with 16 filters
    return SegResNet(blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=cfg["init_filters"],
                     in_channels=4, out_channels=3, dropout_prob=cfg["dropout"])


class DiceBCELoss(torch.nn.Module):
    """Dice + binary cross-entropy, one sigmoid per region.

    Not MONAI's DiceCELoss: with 3 channels it uses softmax cross-entropy, which assumes one class per voxel.
    Our regions overlap (ET is inside TC, TC inside WT), so each channel is its own yes/no.
    """

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, target) + self.bce(logits, target)


def predict_volume(model: torch.nn.Module, image: torch.Tensor, patch: tuple[int, int, int],
                   sw_batch_size: int = 4, overlap: float = 0.5) -> torch.Tensor:
    """Whole brain [B, 4, H, W, D] → boolean region masks [B, 3, H, W, D], by overlapping 3D windows."""
    logits = sliding_window_inference(image, patch, sw_batch_size, model, overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits.float()) > 0.5


def region_dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice per region for one case, [3, ...] masks → [3].

    BraTS rule for an empty true region (33 cases have no enhancing tumor): predicting nothing scores 1,
    predicting anything scores 0.
    """
    pred, target = pred.bool().flatten(1), target.bool().flatten(1)
    inter = (pred & target).sum(1).float()
    total = pred.sum(1).float() + target.sum(1).float()
    return torch.where(total == 0, torch.ones_like(total), 2 * inter / total.clamp(min=1))
