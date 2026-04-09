import logging
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import groq
import os
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class VariableType(str, Enum):
    NORMINAL = "norminal"
    ORDINAL = "ordinal"

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, nomnal_columns: list):
        self.nomnal_columns = nomnal_columns
        self.encoder_dict = {}
        os.makedirs('artifacts/encode', exist_ok=True)

    def encode(self, df,encoder_path = os.path.join('artifacts/encode')) -> pd.DataFrame:
        for column in self.nomnal_columns:
            unique_values = df[column].unique()
            encoder_dict = {value: idx for idx, value in enumerate(unique_values)}
            self.encoder_dict[column] = encoder_dict

            encoder_path = os.path.join('artifacts/encode',f"{column}_encoder.json")
            with open(encoder_path, 'w') as f:
                json.dump(encoder_dict, f)  
            df[column] = df[column].map(encoder_dict)
           
        return df

    def get_encoder_dict(self):
        return self.encoder_dict

class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings: list):
        self.ordinal_mappings = ordinal_mappings
        

    def encode(self, df) :
        for column, mapping in self.ordinal_mappings.items():
            df[column] = df[column].map(mapping)
            logging.info(f"Encoded column '{column}' using ordinal encoding with mapping: {mapping}")   
           
        return df