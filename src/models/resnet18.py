"""ImageNet ResNet-18 with one raw AI logit per image."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class BinaryResNet18(nn.Module):
    """Train layer4 and the binary head; keep all BatchNorm statistics fixed.

    ``weights=None`` is reserved for tests or explicit random-init experiments.
    Loading a checkpoint does not need to download ImageNet weights again.
    """

    def __init__(self, weights="IMAGENET1K_V1"):
        super().__init__()
        if weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("weights must be IMAGENET1K_V1 or null")
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if weights else None
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        for name, param in self.backbone.named_parameters():
            param.requires_grad_(name.startswith(("layer4.", "fc.")))
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False alone does not freeze running_mean/running_var.
        for layer in self.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def forward(self, images):
        return self.backbone(images).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, images):
        if self.training:
            raise RuntimeError("call model.eval() before predict_proba")
        return self(images).sigmoid()
