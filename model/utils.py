# model/utils.py

import joblib
import os


def save_model(model, filename):
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, f"saved_models/{filename}")