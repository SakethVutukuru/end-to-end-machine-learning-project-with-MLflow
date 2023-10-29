import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib')) #load the model

    
    def predict(self, data): #data is the data from the user
        prediction = self.model.predict(data)

        return prediction