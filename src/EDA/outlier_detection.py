import logging
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'   )   
load_dotenv()

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class IQROutlierStrategy(OutlierDetectionStrategy):

    def detect_outliers(self, df, columns):
        outliers=pd.DataFrame(False, index=df.index, columns=columns)
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = (df[column] < lower_bound) | (df[column] > upper_bound)
            n_outliers = outliers[column].sum()
            logging.info(f'Detected {n_outliers} outliers in column: {column}')
        return outliers

class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect(self, df: pd.DataFrame, selected_columns: list) -> pd.DataFrame:
        return self.strategy.detect_outliers(df, selected_columns)

    def handle_outliers(self, df, selected_columns, method: str = 'remove') -> pd.DataFrame:
        outliers = self.detect(df, selected_columns)
        outlier_count = outliers.sum(axis=1)
        rows_to_remove = outlier_count >= 2
        return df[~rows_to_remove]