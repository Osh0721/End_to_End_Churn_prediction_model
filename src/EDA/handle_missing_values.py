import logging
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'   )   
load_dotenv()


class MissingValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        pass

class DropMissingValuesStrategy(MissingValueStrategy):
    def __init__(self, critical_columns = []):
        self.critical_columns = critical_columns
        logging.info(f'Dropping row with missing values in columns: {self.critical_columns}')

    def handle(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        df_cleaned = df.dropna(subset=[self.critical_columns])
        n_dropped = len(df) - len(df_cleaned)
        logging.info(f'Dropped {n_dropped} rows with missing values in columns: {self.critical_columns}')
        return df_cleaned

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class GenderPredication(BaseModel):
    firstname: str
    lastname: str
    pred_gender: Gender

class GenderImputer:
    def __init__(self, model_path: str):
       self.groq_cleint = groq.Groq()

    def _predict_gender(self, firstname: str, lastname: str) -> pd.DataFrame:
       
        prompt = f"""
        
        what is the most likley gender (Male or Female) for someone with the first name {firstname} and last name {lastname}?
        
        Your repsonse only consisit of one word : Male or Female
        """
        repsonse = groq.chat.completions.create(
                        model = "llama-3.3-70b-versatile",
                        messages = [{
                            "role": "user",
                            "content": prompt
                        }])

        response = repsonse.choices[0].message.content.strip() 
        pridection = GenderPredication(firstname=firstname, lastname=lastname, pred_gender=response)
        logging.info(f'Predicted gender for {firstname} {lastname}: {response}')
        return pridection.pred_gender

    def impute(self,df):
        missing_gender_index = df_impute['Gender'].isnull()

        for index in df_impute[missing_gender_index].index:
            firstname = df_impute.loc[index, "Firstname"]
            lastname = df_impute.loc[index, "Lastname"]
            gender = self._predict_gender(firstname, lastname)
            if gender:
                df_impute.loc[index,'Gender']  = gender
                print(f"firstname: {firstname}, lastname: {lastname}, gender: {gender}")
            else:
                print(f"{firstname } {lastname}: No Gender Predicted")

        return df_impute

class FillMissingValuesStrategy(MissingValueStrategy):
   """
   Missing -> Mean(Age)
   """

   def __init__(self, method = 'mean', relevant_columns = None, is_custom_impute=False, custom_imputer = None):
        self.method = method
        self.relevant_columns = relevant_columns
        self.is_custom_impute = is_custom_impute
        self.custom_imputer = custom_imputer

   def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.is_custom_impute:
            logging.info(f'Using custom imputer: {self.custom_imputer.__class__.__name__}')
            return self.custom_imputer.impute(df)
            
        df[self.relevant_columns] = df[self.relevant_columns].fillna(df[self.relevant_columns].mean())
        logging.info(f'Filled missing values in columns: {self.relevant_columns} using method')
        return df