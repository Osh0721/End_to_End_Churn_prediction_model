
import json
import logging
import os
import joblib, sys
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from EDA.feature_binning import CustomBinningStrategy
from EDA.feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from EDA.feature_scaling import MinMaxScalingStrategy
from Model_processing.model_inferance import ModelInference
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config, get_scaling_config
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def streaming_inference_pipeline(data):
    inference = ModelInference(model_path=os.path.join('artifacts', 'models', 'Churn_model.joblib'))
    inference.load_encoders(encoders_dir=os.path.join('artifacts', 'encoders'))
    inference.load_model()  
    result = inference.predict(data)
    return result


if __name__ == "__main__":
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
    result = streaming_inference_pipeline(data)
    print(result)