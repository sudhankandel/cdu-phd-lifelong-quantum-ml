
"""
Architecture:
    Image -> Frozen ResNet-18 -> 512 features
          -> Linear(512, 6) -> Class logits

The model uses a single six-class classification head
throughout all three sequential learning tasks.

The backbone is frozen to support a controlled comparison
with the hybrid quantum-classical model.
"""

import torch
import torch.nn as nn

from torchvision import models, transforms

from config import (
    SEED,
    NUM_CLASSES,
    IMAGE_SIZE,
    get_device,
    set_seed,
)


# --------------------------------------------------
# 1. Image preprocessing
# --------------------------------------------------

def get_transforms():
    """
    Deterministic preprocessing for pretrained ResNet-18.

    The same preprocessing should be used for both
    classical and quantum models to ensure fairness.
    """

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# --------------------------------------------------
# 2. Classical model
# --------------------------------------------------

class ClassicalResNet18(nn.Module):
    """
    Pretrained ResNet-18 with a frozen feature extractor
    and a trainable six-class linear classification head.

    Args:
        num_classes: Number of output classes.
        pretrained: Load pretrained ImageNet weights.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
    ):
        super().__init__()

        if num_classes != 6:
            raise ValueError(
                "This assessment requires six classes."
            )

        # Load pretrained ImageNet weights
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = models.resnet18(weights=weights)

        # Remove original ImageNet classifier
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # Freeze all backbone parameters
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.backbone = backbone
        self.feature_dim = feature_dim

        # Single six-class classification head
        self.classifier = nn.Linear(
            in_features=feature_dim,
            out_features=num_classes,
        )

        # Keep backbone in evaluation mode
        # even when training the classifier
        self.backbone.eval()

    def train(self, mode=True):
        """
        Override train() so frozen backbone BatchNorm
        running statistics remain unchanged.
        """

        super().train(mode)
        self.backbone.eval()

        return self

    def extract_features(self, images):
        """
        Extract 512-dimensional features.

        Backbone gradients are disabled because
        its parameters are frozen.
        """

        with torch.no_grad():
            features = self.backbone(images)

        return features

    def forward(self, images):
        """
        Args:
            images: Tensor [batch_size, 3, 224, 224]

        Returns:
            logits: Tensor [batch_size, 6]
        """

        features = self.extract_features(images)
        logits = self.classifier(features)

        return logits


# --------------------------------------------------
# 3. Parameter counting
# --------------------------------------------------

def count_parameters(model):
    """Return total and trainable parameter counts."""

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# --------------------------------------------------
# 4. Model factory
# --------------------------------------------------

def create_classical_model(
    pretrained=True,
    device=None,
):
    """
    Create a reproducibly initialised classical model.

    The seed is set before constructing the classifier,
    ensuring consistent initial head parameters.
    """

    set_seed(SEED)

    if device is None:
        device = get_device()

    model = ClassicalResNet18(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
    )

    model = model.to(device)

    return model


# --------------------------------------------------
# 5. Basic model verification
# --------------------------------------------------

def main():

    device = get_device()

    print("=" * 50)
    print("CDU PhD - Classical ResNet-18 Baseline")
    print("=" * 50)

    print("Seed:", SEED)
    print("Device:", device)

    model = create_classical_model(
        pretrained=True,
        device=device,
    )

    total, trainable = count_parameters(model)

    print("\nArchitecture:")
    print("Backbone: ResNet-18")
    print("Feature dimension:", model.feature_dim)
    print("Output classes:", NUM_CLASSES)

    print("\nParameters:")
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen: {total - trainable:,}")

    # Dummy input for architecture verification
    dummy_images = torch.randn(
        2,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        output = model(dummy_images)

    print("\nInput shape:", dummy_images.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, NUM_CLASSES)
    assert trainable == 3078

    print("\nModel verification successful!")


if __name__ == "__main__":
    main()
