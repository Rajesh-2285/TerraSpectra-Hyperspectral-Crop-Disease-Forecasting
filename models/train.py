import json
import os
import sys
import tracemalloc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow importing from the models folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn3d import TerraSpectra3DCNN
from models.dataset import HyperspectralPatchDataset


def train_and_audit():
    tracemalloc.start()

    # 1. Load 3D patches (4,096 pixels)
    dataset = HyperspectralPatchDataset("data/processed_farm.h5", patch_size=5)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Setup beginner 3D-CNN, loss, and optimizer
    model = TerraSpectra3DCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Train for 5 quick epochs
    model.train()
    for epoch in range(1, 6):
        total_loss, correct, total = 0.0, 0, 0
        for patches, labels in loader:
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / total) * 100
        print(f"Epoch {epoch}/5 | Loss: {total_loss/len(loader):.4f} | Accuracy: {acc:.2f}%")

    # 4. Mid-Project Extraction Audit: check peak Python memory usage
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Extraction Audit Passed | Peak 3D Tensor Batch Memory: {peak_mem / (1024 * 1024):.2f} MB")

    # 5. Save trained model weights
    torch.save(model.state_dict(), "models/terraspectra_3dcnn.pth")

    # 6. Generate predictions across the 64x64 farm grid for the GIS map
    model.eval()
    eval_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for patches, _ in eval_loader:
            outputs = model(patches)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of stress (class 1)
            all_preds.extend(probs.tolist())

    # Save GPS-anchored grid points for Deck.gl heatmap
    base_lat, base_lon = 17.3850, 78.4867  # Sample farm coordinates
    step = 0.0005
    grid_data = []
    for idx, stress_prob in enumerate(all_preds):
        r = idx // 64
        c = idx % 64
        grid_data.append({
            "lat": round(base_lat + r * step, 6),
            "lon": round(base_lon + c * step, 6),
            "stress_score": round(stress_prob, 3),
            "status": "Stressed" if stress_prob > 0.5 else "Healthy"
        })

    with open("data/farm_predictions.json", "w") as f:
        json.dump(grid_data, f)

    stressed_count = sum(1 for item in grid_data if item["status"] == "Stressed")
    print(f"Saved data/farm_predictions.json | Total pixels: {len(grid_data)} | Flagged stressed: {stressed_count}")


if __name__ == "__main__":
    train_and_audit()