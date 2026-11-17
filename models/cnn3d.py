import torch
import torch.nn as nn


class TerraSpectra3DCNN(nn.Module):
    """3D Convolutional Neural Network for spatial-spectral crop stress classification."""

    def __init__(self, num_classes=2):
        super().__init__()
        # Input shape: [B, 1, 16, 5, 5] -> (Batch, Channels, Depth, Height, Width)
        self.features = nn.Sequential(
            # Conv3d 1: [B, 1, 16, 5, 5] -> [B, 8, 16, 5, 5]
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            # MaxPool3d: [B, 8, 16, 5, 5] -> [B, 8, 8, 5, 5]
            nn.MaxPool3d(kernel_size=(2, 1, 1)),
            # Conv3d 2: [B, 8, 8, 5, 5] -> [B, 16, 8, 5, 5]
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            # AdaptiveAvgPool3d: [B, 16, 8, 5, 5] -> [B, 16, 1, 1, 1]
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        # Classifier: [B, 16] -> [B, num_classes]
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to [B, 16]
        return self.classifier(x)

    def count_parameters(self):
        """Returns total number of trainable parameters in the 3D-CNN."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TerraSpectra3DCNN()
    sample_batch = torch.randn(4, 1, 16, 5, 5)
    out = model(sample_batch)
    print(
        f"Model test passed | Input: {tuple(sample_batch.shape)} -> Output: {tuple(out.shape)} | "
        f"Trainable params: {model.count_parameters()}"
    )