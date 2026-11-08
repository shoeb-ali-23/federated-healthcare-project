import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"

# Network & Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 7860))
API_KEY = os.getenv("API_KEY", "DEV_SECRET_KEY_PLACEHOLDER")

# Federated Learning Hyperparameters
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COLUMN = "condition"

DEFAULT_ROUNDS = 5
DEFAULT_LOCAL_EPOCHS = 3
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPSILON = 1.0

