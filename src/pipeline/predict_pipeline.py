import sys 
import os 
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object  #to load pickle file

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'            #responsible for handling categorical features, feature scaling etc
            model = load_object(file_path = model_path)                 #load pickle file, utils.py
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:          #responsible for mapping html frontend inputs to the backend
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,   #not compulsory to provide datatype. Python is smart.
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        

            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score

    def get_data_as_data_frame(self):           #returns input data as dataframe for backend processing
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
