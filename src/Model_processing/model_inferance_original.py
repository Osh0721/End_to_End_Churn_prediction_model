import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.EDA.feature_binning import CustomBinningStrategy
from src.EDA.feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from src.EDA.feature_scaling import MinMaxScalingStrategy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.config import get_binning_config, get_encoding_config, get_scaling_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 


"""

{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88,
  "Exited": 1
}
"""
class ModelInferencer:
    def __init__(self,model_path):
        self.model_path=model_path
        self.model = None
        self.encoder = {}

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise ValueError("Can't load. File not found.")

        self.model = joblib.load(self.model_path)

    
    def load_encoder(self, encoder_path):
        for file in os.listdir(encoder_path):
            if not file.endswith('_encoder.json'):  # skip non-encoder files
                continue
            feature_name = file.replace('_encoder.json', '')  # ← was splitting on wrong string
            with open(os.path.join(encoder_path, file), 'r') as f:
                self.encoder[feature_name] = json.load(f)

           
    def preprocess_input(self, data):
        data=pd.DataFrame([data])

        for col, encoders in self.encoder.items():
           data[col] = data[col].map(encoders)

        binning_config = get_binning_config()
        binning = CustomBinningStrategy(binning_config['credit_score_bins'])
        data = binning.bin_feature(data, 'CreditScore')
        encoding_config = get_encoding_config()
        ordinal_stratgey = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])
        data = ordinal_stratgey.encode(data)

        data=data.drop(columns=['CustomerId', 'RowNumber', 'Lastname', 'Firstname'], errors='ignore')

        return data

    def predict(self, data):
        pp_data = self.preprocess_input(data)
        print("Preprocessed data for inference:", pp_data)
        y_pred = self.model.predict(pp_data)

        y_proba = float(self.model.predict_proba(pp_data)[:, 1][0])

        y_pred_label='Churn' if y_pred[0] == 1 else 'Retain'
        y_prob=round(y_proba * 100, 2)

        return {"prediction": y_pred_label, "probability": y_prob}

data={
  "RowNumber": 1,
  "CustomerId": 15634602,
  "Firstname": "Grace",
  "Lastname": "Williams",
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88,}

inference = ModelInferencer(model_path=os.path.join('artifacts', 'models', 'Churn_model.joblib'))
inference.load_encoder(encoder_path=os.path.join('artifacts', 'encoders'))
inference.load_model()  
result = inference.predict(data)
print(result)

