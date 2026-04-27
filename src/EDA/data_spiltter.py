"""
Data splitting strategies for both pandas and PySpark DataFrames.
Students can compare train-test split implementations between pandas and PySpark.
"""

import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd  # Keep for educational comparison
from sklearn.model_selection import train_test_split  # For pandas implementation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from Spark.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies."""
    
    ############### PANDAS CODES ###########################
    # @abstractmethod
    # def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    #     pass
    
    ############### PYSPARK CODES ###########################
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy."""
    
    ############### PANDAS CODES ###########################
    # def __init__(self, test_size = 0.2):
    #     self.test_size= test_size
    #     logger.info(f"SimpleTrainTestSplitStratergy initialized with test_size={test_size}")
    #
    # def split_data(self, df, target_column):
    # 
    #     Y = df[target_column]
    #     X = df.drop(columns=[target_column])
    #     
    #     # Log target distribution
    #     target_dist = Y.value_counts()
    #     logger.info(f"\nTarget Variable Distribution:")
    #     for value, count in target_dist.items():
    #         logger.info(f"  {value}: {count} ({count/len(Y)*100:.2f}%)")
    #   
    #     # Perform split
    #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)
    #   
    #     logger.info(f"\n{'='*60}")
    #     logger.info(f"✓ DATA SPLITTING COMPLETE")
    #     logger.info(f"{'='*60}\n")
    #     return X_train, X_test, Y_train, Y_test
    
    ############### PYSPARK CODES ###########################
    def __init__(self, test_size: float = 0.2, random_seed: int = 42, spark: Optional[SparkSession] = None):
        """
        Initialize simple train-test split strategy.
        
        Args:
            test_size: Proportion of data for test set (0-1)
            random_seed: Random seed for reproducibility
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_seed = random_seed
        logger.info(f"SimpleTrainTestSplitStratergy initialized with test_size={test_size}")
    
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Perform simple random train-test split.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMPLE TRAIN-TEST SPLIT (PySpark)")
        logger.info(f"{'='*60}")
        logger.info(f"Starting train-test split with test_size={self.test_size}")
        
        # Split into train and test
        train_df, test_df = df.randomSplit([1 - self.test_size, self.test_size], seed=self.random_seed)
        
        # Separate features and target
        feature_columns = [col for col in train_df.columns if col != target_column]
        
        X_train = train_df.select(feature_columns)
        Y_train = train_df.select(target_column)
        X_test = test_df.select(feature_columns)
        Y_test = test_df.select(target_column)
        
        # Log split information
        train_count = X_train.count()
        test_count = X_test.count()
        total_count = train_count + test_count
        
        logger.info(f"\nSplit Results:")
        logger.info(f"  • Total samples: {total_count}")
        logger.info(f"  • Train samples: {train_count} ({train_count/total_count*100:.1f}%)")
        logger.info(f"  • Test samples: {test_count} ({test_count/total_count*100:.1f}%)")
        
        # Log target distribution
        logger.info(f"\nTarget Variable Distribution:")
        train_dist = Y_train.groupBy(target_column).count().collect()
        for row in train_dist:
            logger.info(f"  Train - {row[target_column]}: {row['count']} samples")
        
        test_dist = Y_test.groupBy(target_column).count().collect()
        for row in test_dist:
            logger.info(f"  Test - {row[target_column]}: {row['count']} samples")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ DATA SPLITTING COMPLETE")
        logger.info(f"{'='*60}\n")
        
        return X_train, X_test, Y_train, Y_test


class DataSplitter:
    """Main data splitter class that uses different strategies."""
    
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialize data splitter with a specific strategy.
        
        Args:
            strategy: DataSplittingStrategy instance
        """
        self.strategy = strategy
        logger.info(f"DataSplitter initialized with strategy: {strategy.__class__.__name__}")
    
    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data using the configured strategy.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        return self.strategy.split_data(df, target_column)

def create_simple_splitter(test_size: float = 0.2, spark: Optional[SparkSession] = None):
    """Create a simple train-test splitter (backward compatibility)."""
    return SimpleTrainTestSplitStratergy(test_size=test_size, spark=spark)