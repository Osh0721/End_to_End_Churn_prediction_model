"""
Feature scaling strategies for PySpark DataFrames.
Supports MinMaxScaler and StandardScaler transformations.
"""

import logging
import os
import json
from enum import Enum
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from Spark.spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureScalingStrategy(ABC):
    """Abstract base class for feature scaling strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
        self.fitted_model = None
    
    @abstractmethod
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Scale specified columns in the DataFrame.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled features
        """
        pass


class ScalingType(str, Enum):
    """Enumeration of scaling types."""
    MINMAX = 'minmax'
    STANDARD = 'standard'


class MinMaxScalingStrategy(FeatureScalingStrategy):
    """Min-Max scaling strategy to scale features to [0, 1] range."""
    
    def __init__(self, output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Min-Max scaling strategy.
        
        Args:
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        self.pipeline_model = None
        self.columns_to_scale = None
        logger.info("MinMaxScalingStrategy initialized (PySpark)")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Min-Max scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        self.columns_to_scale = columns_to_scale
        df_scaled = df
        
        logger.info(f"Applying MinMax scaling to columns: {columns_to_scale}")
        
        # Build a single pipeline for all columns
        stages = []
        
        # Create vector assembler for all columns
        assembler = VectorAssembler(
            inputCols=columns_to_scale,
            outputCol="features_to_scale"
        )
        stages.append(assembler)
        
        # Create MinMaxScaler
        scaler = MinMaxScaler(
            inputCol="features_to_scale",
            outputCol="scaled_features"
        )
        stages.append(scaler)
        
        # Create and fit pipeline
        pipeline = Pipeline(stages=stages)
        self.pipeline_model = pipeline.fit(df_scaled)
        
        # Transform data
        df_scaled = self.pipeline_model.transform(df_scaled)
        
        # Extract scaled values back to original columns
        # Use UDF to extract values from vector
        from pyspark.ml.linalg import Vectors, VectorUDT
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        # Log scaling statistics
        scaler_model = self.pipeline_model.stages[1]
        for i, col in enumerate(columns_to_scale):
            min_val = float(scaler_model.originalMin[i])
            max_val = float(scaler_model.originalMax[i])
            logger.info(f"✓ Scaled '{col}': min={min_val:.4f}, max={max_val:.4f}")
        
        return df_scaled
    
    def save_scaler(self, columns_to_scale: List[str], save_dir: str = 'artifacts/scale') -> bool:
        """
        Save the fitted scaler model and metadata for inference.
        
        Args:
            columns_to_scale: List of columns that were scaled
            save_dir: Directory to save scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            if self.pipeline_model is None:
                logger.error("✗ No fitted scaler model to save")
                return False
            
            # Save the pipeline model
            model_path = os.path.join(save_dir, 'minmax_scaler_pipeline')
            self.pipeline_model.write().overwrite().save(model_path)
            logger.info(f"✓ Saved PySpark scaler pipeline to: {model_path}")
            
            # Extract and save metadata for compatibility
            scaler_model = self.pipeline_model.stages[1]  # MinMaxScaler is second stage
            
            # Convert Spark vectors to lists for JSON serialization
            metadata = {
                'columns_to_scale': columns_to_scale,
                'data_min': [float(x) for x in scaler_model.originalMin.toArray()],
                'data_max': [float(x) for x in scaler_model.originalMax.toArray()],
                'n_features': len(columns_to_scale),
                'scaling_type': 'minmax',
                'framework': 'pyspark'
            }
            
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Saved scaling metadata to: {metadata_path}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to save scaler: {str(e)}")
            return False
    
    def load_scaler(self, save_dir: str = 'artifacts/scale') -> bool:
        """
        Load the fitted scaler model for inference.
        
        Args:
            save_dir: Directory containing scaler artifacts
            
        Returns:
            bool: True if successful
        """
        try:
            model_path = os.path.join(save_dir, 'minmax_scaler_pipeline')
            metadata_path = os.path.join(save_dir, 'scaling_metadata.json')
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logger.error(f"✗ Scaler artifacts not found in: {save_dir}")
                return False
            
            # Load pipeline model
            self.pipeline_model = PipelineModel.load(model_path)
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.columns_to_scale = metadata['columns_to_scale']
            
            self.fitted_model = self.pipeline_model
            logger.info(f"✓ Loaded scaler from: {save_dir}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {str(e)}")
            return False
    
    def transform(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply the loaded scaler to transform data (no fitting).
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        if self.pipeline_model is None:
            raise ValueError("Scaler not loaded/fitted. Call load_scaler() or scale() first.")
        
        # Transform using the loaded pipeline
        df_scaled = self.pipeline_model.transform(df)
        
        # Extract scaled values back to original columns
        # Use UDF to extract values from vector
        from pyspark.ml.linalg import Vectors, VectorUDT
        from pyspark.sql.types import DoubleType
        
        def get_vector_element(idx):
            def extract(vector):
                return float(vector[idx]) if vector is not None else None
            return F.udf(extract, DoubleType())
        
        for i, col in enumerate(columns_to_scale):
            df_scaled = df_scaled.withColumn(
                col,
                get_vector_element(i)(F.col("scaled_features"))
            )
        
        # Drop intermediate columns
        df_scaled = df_scaled.drop("features_to_scale", "scaled_features")
        
        return df_scaled


class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling strategy to scale features to zero mean and unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, 
                 output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        """
        Initialize Standard scaling strategy.
        
        Args:
            with_mean: Whether to center the data before scaling
            with_std: Whether to scale the data to unit variance
            output_col_suffix: Suffix to add to scaled column names
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
        logger.info(f"StandardScalingStrategy initialized (PySpark) - "
                   f"with_mean={with_mean}, with_std={with_std}")
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        """
        Apply Standard scaling to specified columns.
        
        Args:
            df: PySpark DataFrame
            columns_to_scale: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        df_scaled = df 

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            assembler = VectorAssembler(inputCols=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = StandardScaler(inputCol=vector_col, outputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            df_scaled = df_scaled.withColumn(
                                            col,
                                            get_value_udf(F.col(scaled_vector_col))
                                            )

        return df_scaled