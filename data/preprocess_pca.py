import h5py
import numpy as np
from sklearn.decomposition import PCA


def run_pca_reduction(n_components=16):
    # 1. Load the raw 200-band hyperspectral cube
    with h5py.File("data/mock_farm.h5", "r") as f:
        cube = f["hyperspectral_cube"][:]
        labels = f["labels"][:]

    height, width, bands = cube.shape

    # 2. Flatten spatial pixels into (4096 pixels, 200 bands) for PCA
    flat_pixels = cube.reshape(-1, bands)

    # 3. Compress 200 bands -> 16 principal components
    pca = PCA(n_components=n_components, random_state=42)
    reduced_pixels = pca.fit_transform(flat_pixels)

    # 4. Reshape back into a 3D cube: (64, 64, 16)
    reduced_cube = reduced_pixels.reshape(height, width, n_components).astype(
        np.float32
    )
    explained = pca.explained_variance_ratio_.sum() * 100

    # 5. Save the compressed cube
    output_path = "data/processed_farm.h5"
    with h5py.File(output_path, "w") as f:
        f.create_dataset("pca_cube", data=reduced_cube)
        f.create_dataset("labels", data=labels)

    print(
        f"Saved {output_path} | New shape: {reduced_cube.shape} | Variance retained: {explained:.2f}%"
    )


if __name__ == "__main__":
    run_pca_reduction()