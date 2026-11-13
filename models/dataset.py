import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HyperspectralPatchDataset(Dataset):
    def __init__(self, h5_path="data/processed_farm.h5", patch_size=5):
        with h5py.File(h5_path, "r") as f:
            cube = f["pca_cube"][:]      # Shape: (64, 64, 16)
            labels = f["labels"][:]      # Shape: (64, 64)

        self.patch_size = patch_size
        pad = patch_size // 2

        # Pad spatial edges so border pixels can also form 5x5 patches
        padded_cube = np.pad(
            cube, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
        )

        self.patches = []
        self.targets = []

        height, width, _ = cube.shape
        for r in range(height):
            for c in range(width):
                patch = padded_cube[r : r + patch_size, c : c + patch_size, :]
                # PyTorch Conv3d expects (Channels=1, Depth=Bands, Height, Width)
                patch = np.transpose(patch, (2, 0, 1))[np.newaxis, ...]
                self.patches.append(patch)
                self.targets.append(labels[r, c])

        self.patches = np.array(self.patches, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx]), torch.tensor(self.targets[idx])


if __name__ == "__main__":
    ds = HyperspectralPatchDataset()
    x, y = ds[0]
    print(f"Total 3D patches: {len(ds)} | Single patch tensor shape: {tuple(x.shape)} | Label: {y.item()}")