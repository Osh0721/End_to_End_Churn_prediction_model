import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')  

class FeatureBinningStrategy(ABC):
    @abstractmethod
    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass    

class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bins_definitions: dict):
        self.bins = bins_definitions.get('bins')

    def bin_feature(self, df: pd.DataFrame, column: str) -> pd.DataFrame:

        def assign_bin(value):
            for bin_label,bin_range in self.bins_definitions.items():
                if len(bin_range) == 2:
                 if bin_range[0]<= value <= bin_range[1]:
                    return bin_label    

            return 'Invalid'

        df[column + '_binned'] = df[column].apply(assign_bin)
        del df[column]
        logging.info(f'Binned column: {column} into {len(self.bins)} bins')
        return df
