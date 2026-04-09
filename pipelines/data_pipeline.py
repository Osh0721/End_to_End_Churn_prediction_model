import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from EDA.data_ingestion import DataIngestorCSV
from EDA.handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from EDA.outlier_detection import OutlierDetector, IQROutlierDetection
from EDA.feature_binning import CustomBinningStratergy
from EDA.feature_encoding import OrdinalEncodingStratergy, NominalEncodingStrategy
from EDA.feature_scaling import MinMaxScalingStratergy
from EDA.data_spiltter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config

def data_pipeline(
    data_path: str = 'data/raw/ChurnModelling.csv',
    target_column: str = 'Exited',
    test_size: float = 0.2,
    force_rebuild: bool = False
) -> Dict[str, np.ndarray]:

    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()
    missing_values_config = get_missing_values_config()

    print("Starting data pipeline... Step 1: Data Ingestion")
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths[data_artifacts_dir])
    x_train_path = os.path.join('artifacts', 'data', 'X_train.csv')
    x_test_path = os.path.join('artifacts', 'data', 'X_test.csv')
    y_train_path = os.path.join('artifacts', 'data', 'Y_train.csv')
    y_test_path = os.path.join('artifacts', 'data', 'Y_test.csv')

    if os.path.exists(x_train_path) and os.path.exists(x_test_path) and os.path.exists(y_train_path) and os.path.exists(y_test_path) and not force_rebuild:
        logger.info("Artifacts already exist. Loading from disk...")
        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path)
        y_test = pd.read_csv(y_test_path)

    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)
    print("loaded data with shape:", df.shape)




