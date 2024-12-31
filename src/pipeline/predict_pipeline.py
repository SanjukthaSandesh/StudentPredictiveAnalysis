import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            print("Starting prediction...")
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("Transforming data...")
            data_scaled = preprocessor.transform(features)

            print("Making predictions...")
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender: str, CGPA100: float, CGPA200: float, CGPA300: float, CGPA400: float, SGPA: float):
        self.gender = gender
        self.CGPA100 = CGPA100
        self.CGPA200 = CGPA200
        self.CGPA300 = CGPA300
        self.CGPA400 = CGPA400
        self.SGPA = SGPA

    def get_data_as_data_frame(self):
        try:
            data = {
                'gender': [self.gender],
                'CGPA100': [self.CGPA100],
                'CGPA200': [self.CGPA200],
                'CGPA300': [self.CGPA300],
                'CGPA400': [self.CGPA400],
                'SGPA': [self.SGPA],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
