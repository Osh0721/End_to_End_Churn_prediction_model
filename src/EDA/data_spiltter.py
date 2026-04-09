import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DaraSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
        pass

class splitType(str, Enum):
    SIMPLE = "simple"
    STRATIFIED = "stratified"

class SimpleDataSplittingStrategy(DaraSplittingStrategy):
    def __init__(self,test_size=0.2):
        self.test_size = test_size

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
        X = df.drop(columns=[target_column])
        y = df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        logging.info(f"Performed simple data splitting with test size {self.test_size} and random state {random_state}.")
        return X_train, X_test, y_train, y_test