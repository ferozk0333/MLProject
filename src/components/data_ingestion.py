# All the code relevant to reading the data from different sources
# Aim: Read data from data source and split data into train_test

#importing essential modules
import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass                       # used to create class variables

from src.components.data_transformation import DataTransformation, DataTransformationConfig

# any input that is required will be passed to this class
@dataclass                                              # decorator - allows to directly define class variables without __init__
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")     # all outputs will be saved in artifact folder
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()               # as soon as DataIngestion class is called, the above 3 objects will get stored in ingestion_config variable

    def initiate_data_ingestion(self):                              # code to read data from data source
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')              # reading from local data source
            
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)    #creating folder

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state= 32)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)       # saving in artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path                         # returns path to train/test set for next step
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj  = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()            #creating an object/initializing
    data_transformation.initiate_data_transformation(train_data,test_data)    #initiating a method of this object
   
            


