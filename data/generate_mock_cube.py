import h5py
import numpy as np


def create_mock_farm():
    # 1. Create a 64x64 pixel farm with 200 hyperspectral light bands
    height, width, bands = 64, 64, 200
    np.random.seed(42)

    # Simulate healthy green crop spectral reflectance (values between 0.2 and 0.8)
    cube = np.random.normal(loc=0.6, scale=0.05, size=(height, width, bands)).astype(
        np.float32
    )

    # 2. Create labels: 0 = Healthy crop, 1 = Chemically stressed (early fungal blight)
    labels = np.zeros((height, width), dtype=np.int64)

    # Inject a stressed zone in the center-right (rows 20 to 35, cols 30 to 45)
    labels[20:35, 30:45] = 1
    # Alter invisible chlorophyll/NIR bands (bands 80 to 150) for the stressed zone
    cube[20:35, 30:45, 80:150] -= 0.25

    # 3. Save as an HDF5 hyperspectral cube
    output_path = "data/mock_farm.h5"
    with h5py.File(output_path, "w") as f:
        f.create_dataset("hyperspectral_cube", data=cube)
        f.create_dataset("labels", data=labels)

    print(
        f"Saved {output_path} | Cube shape: {cube.shape} | Stressed pixels: {labels.sum()}"
    )


if __name__ == "__main__":
    create_mock_farm()