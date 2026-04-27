import logging
import pandas as pd
from abc import ABC, abstractmethod
import groq
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'   )   


class ModelEvaluator():
    def __init__(self,model,model_name):
        self.model = model
        self.model_name = model_name
        self.evaluation_results = {}
    
    def evaluate(self, X_test, y_test):
        logging.info(f"Evaluating model: {self.model_name}")
        y_pred = self.model.predict(X_test)
        # self.evaluation_results['accuracy'] = accuracy_score(y_test, y_pred)
        # self.evaluation_results['precision'] = precision_score(y_test, y_pred)
        # self.evaluation_results['recall'] = recall_score(y_test, y_pred)
        # self.evaluation_results['f1_score'] = f1_score(y_test, y_pred)
        # self.evaluation_results['classification_report'] = classification_report(y_test, y_pred)
        # self.evaluation_results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        cm=confusion_matrix(y_test,y_pred)
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        self.evaluation_results = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        return self.evaluation_results
