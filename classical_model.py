import math

import torch
import torch.nn as nn
from torchvision import models, transforms

from config import SEED, NUM_CLASSES, IMAGE_SIZE, get_device, set_seed


def get_transforms():
    """Deterministic preprocessing shared by all models."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ClassicalResNet18(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        if num_classes != 6:
            raise ValueError("This assessment requires six classes.")

        # Pinned weights (not the changeable DEFAULT alias).
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # keep frozen BatchNorm statistics
        return self

    def extract_features(self, images):
        with torch.no_grad():
            return self.backbone(images)

    def forward(self, images):
        return self.classifier(self.extract_features(images))


class MatchedClassicalHead(nn.Module):
    """Classical head with the same trainable-parameter count as the VQC head.

    Mirrors the hybrid structure: 512 -> 6 bounded 'angles' -> 6x6 mixing.
    Linear(512,6) = 3,078 ; Linear(6,6,bias=False) = 36 ; total = 3,114.
    """

    def __init__(self, in_features=512, width=6, num_classes=6):
        super().__init__()
        self.projection = nn.Linear(in_features, width)
        self.mix = nn.Linear(width, num_classes, bias=False)

    def forward(self, x):
        return self.mix(math.pi * torch.tanh(self.projection(x)))


class MLPHead(nn.Module):
    """Higher-capacity classical reference head."""

    def __init__(self, in_features=512, hidden=256, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_classical_model(pretrained=True, device=None):
    set_seed(SEED)
    if device is None:
        device = get_device()
    return ClassicalResNet18(NUM_CLASSES, pretrained).to(device)


def main():
    device = get_device()
    print("Seed:", SEED, "| Device:", device)

    linear = nn.Linear(512, 6)
    matched = MatchedClassicalHead()
    mlp = MLPHead()
    for name, m in [("linear", linear), ("matched", matched), ("mlp", mlp)]:
        print(f"{name:8s} trainable params: {count_parameters(m)[1]:,}")

    assert count_parameters(linear)[1] == 3078
    assert count_parameters(matched)[1] == 3114   # == quantum head
    assert matched(torch.randn(4, 512)).shape == (4, 6)
    assert mlp(torch.randn(4, 512)).shape == (4, 6)

    model = create_classical_model(pretrained=True, device=device)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    assert out.shape == (2, NUM_CLASSES)
    print("Model verification successful!")


if __name__ == "__main__":
    main()
