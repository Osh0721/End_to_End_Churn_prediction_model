import os
import joblib
import logging
import pandas as pd
from typing import Any, Dict, Optional
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from Model_processing.model_buildiing import RandomForestModelBuilder, XGboostModelBuilder
from Model_processing.model_training import ModelTrainer
from data_pipeline import data_pipeline
from config import get_data_paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def training_pipeline(
    data_path: str = "/Users/oshanrathnayaka/Oshan's Personal/End-to-End ML system/Week5_6/End_to_End_Churn_prediction_model/data/raw/CEHHbInToW.csv",
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'models', 'final_model.joblib'),
    ):

    if (
        not os.path.exists(get_data_paths()["X_train"])
        or not os.path.exists(get_data_paths()["X_test"])
        or not os.path.exists(get_data_paths()["Y_train"])
        or not os.path.exists(get_data_paths()["Y_test"])
    ):
        print("Data artifacts not found. Running data pipeline...")
        data_pipeline()
    else:
        print("Data artifacts already exist. Skipping data pipeline...")


training_pipeline()