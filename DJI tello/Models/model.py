import joblib
import numpy as np
model = joblib.load("best_model.pkl")

def predict_side(features){
    return model.predict(features)
}