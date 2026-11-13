import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="TerraSpectra Crop Disease API")

# Allow React frontend to connect cleanly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/farm_predictions.json"


def load_predictions():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


@app.get("/")
def health_check():
    return {"status": "TerraSpectra API is running"}


@app.get("/api/summary")
def get_summary():
    data = load_predictions()
    total = len(data)
    stressed = sum(1 for item in data if item["status"] == "Stressed")
    healthy = total - stressed
    return {
        "total_pixels": total,
        "healthy_pixels": healthy,
        "stressed_pixels": stressed,
        "stress_percentage": round((stressed / total) * 100, 2) if total else 0,
    }


@app.get("/api/predictions")
def get_predictions():
    return load_predictions()