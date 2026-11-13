import torch
import torch.nn as nn


class TerraSpectra3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Input shape: (Batch, 1, 16, 5, 5) -> (Batch, Channels, Depth, Height, Width)
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 1, 1)),  # Reduces depth: 16 -> 8
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),      # Compresses 3D volume to (Batch, 16, 1, 1, 1)
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten to (Batch, 16)
        return self.classifier(x)


if __name__ == "__main__":
    model = TerraSpectra3DCNN()
    sample_batch = torch.randn(4, 1, 16, 5, 5)
    out = model(sample_batch)
    print(f"Model test passed | Input: {tuple(sample_batch.shape)} -> Output: {tuple(out.shape)}")