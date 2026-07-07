# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper Loss.
    输入为B×C×H×W。

    The cosine margin is kept above float32 machine precision. Using 1e-8 as
    the clamp margin can round back to exactly 1.0 in float32, where the
    derivative of acos is infinite and can turn an otherwise finite Stage-3
    update into NaN.
    """

    def __init__(self, eps: float = 1e-8, cosine_margin: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.cosine_margin = float(cosine_margin)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()

        dot = torch.sum(pred * target, dim=1)
        pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        target_norm = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)

        cos = dot / (pred_norm * target_norm + self.eps)
        margin = max(self.cosine_margin, 10.0 * torch.finfo(cos.dtype).eps)
        cos = torch.nan_to_num(
            cos,
            nan=0.0,
            posinf=1.0 - margin,
            neginf=-1.0 + margin,
        )
        cos = torch.clamp(cos, -1.0 + margin, 1.0 - margin)

        angle = torch.acos(cos)
        return torch.mean(angle)
