import joblib
import pandas as pd


class Predictor:

    def __init__(self):

        self.model = joblib.load(
            "models/best_model.pkl"
        )

    def predict(self, input_data):

        prediction = self.model.predict(
            input_data
        )[0]

        probability = self.model.predict_proba(
            input_data
        )[0][1]

        return prediction, probability