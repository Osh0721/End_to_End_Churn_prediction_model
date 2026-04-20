import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from EDA.data_ingestion import DataIngestorCSV
from EDA.handle_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from EDA.outlier_detection import OutlierDetector, IQROutlierStrategy
from EDA.feature_binning import CustomBinningStrategy
from EDA.feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from EDA.feature_scaling import MinMaxScalingStrategy
from EDA.data_spiltter import SimpleDataSplittingStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config



def data_pipeline(
    data_path: str = "/Users/oshanrathnayaka/Oshan's Personal/End-to-End ML system/Week5_6/End_to_End_Churn_prediction_model/data/raw/CEHHbInToW.csv",
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
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
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
        return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
     }
    ingestor = DataIngestorCSV()
    df = ingestor.ingest(data_path)
    print("loaded data with shape:", df.shape)

    print("Step 2: Handling Missing Values")
    drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])
    age_handler = FillMissingValuesStrategy(method='mean',relevant_columns='Age')

    gender_handler = FillMissingValuesStrategy(relevant_columns='Gender',is_custom_impute=True , custom_imputer=GenderImputer())
  
    df = drop_handler.handle(df)
    df = age_handler.handle(df)
    # df = gender_handler.handle(df)
    df.to_csv("temp_imputed.csv", index=False)

  
    print(f"Data shape after handling missing values: {df.shape}")

    print("Step 3: Ourlier Detection")

    outlier_detector = OutlierDetector(strategy=IQROutlierStrategy())
    df = outlier_detector.handle_outliers(df, columns=columns['outlier_columns'])


    print(f"Data shape after handling outliers: {df.shape}")

    print("\nStep 4: Feature Binning ")

    binning = CustomBinningStrategy(bins_definitions=binning_config['credit_score_bins'])
    df = binning.bin_feature(df, column='CreditScore')
    print(f"Data shape after feature binning: {df.shape}")

    print("\nStep 5: Feature Encoding")

    norminal_strategy = NominalEncodingStrategy(encoding_config['nominal_columns'])
    ordinal_strategy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'])
    df= norminal_strategy.encode(df)
    df= ordinal_strategy.encode(df)
    print(f"Data shape after feature encoding: \n{df.head()}")

    print("\nStep 6: Feature Scaling")
    minmax_strategy = MinMaxScalingStrategy()
    df = minmax_strategy.scale(df,scaling_config['columns_to_scale'])
    print(f"Data shape after feature scaling: \n{df.head()}")
    
    print("\nStep 7: Post Processing (Dropping Unnecessary Columns)")
    df.drop(columns=columns['drop_columns'], inplace=True)
    print(f"Data shape after dropping columns: \n{df.head}")

    print("\nStep 8: Data Splitting")
    splitter = SimpleDataSplittingStrategy(test_size=splitting_config['test_size'])
    X_train, X_test, y_train, y_test = splitter.split_data(df,test_size=splitting_config['test_size'], target_column="Exited", random_state=splitting_config['random_state'])

    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)           
    y_test.to_csv(y_test_path, index=False)

data_pipeline()