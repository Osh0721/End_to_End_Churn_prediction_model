import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FeatureScalingStrategy(ABC):
    @abstractmethod
    def scale(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class ScalingMethod(str, Enum):
    MINMAX = "minmax"
    STANDARD = "standard"

class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self):
        self.scaler= MinMaxScaler()
        self.fitted=False
    
    def scale(self,df,column_to_scale):
        df[column_to_scale] = self.scaler.fit_transform(df[column_to_scale])
        self.fitted=True
        logging.info(f"Scaled column '{column_to_scale}' using Min-Max scaling.")
        return df

    def get_scaler(self):
        return self.scaler

class StandardScalingStrategy(FeatureScalingStrategy):
    def __int__(self):
        self.scaler= StandardScaler()
        self.fitted=False
    
    def scale(self,df,column_to_scale):
        df[column_to_scale] = self.scaler.fit_transform(df[[column_to_scale]])
        self.fitted=True
        logging.info(f"Scaled column '{column_to_scale}' using Standard scaling.")
        return df

    def get_scaler(self):
        return self.scaler

