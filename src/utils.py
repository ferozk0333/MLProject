#contains functionalities that are common to applications. Example, read dataset from a database, save model on cloud, etc
import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)           

        os.makedirs(dir_path, exist_ok=True)        #make a directory

        with open(file_path, "wb") as file_obj:     #open the file in write byte mode and we will dill.dump
            dill.dump(obj, file_obj)                #dill helps in creating pickle file
    except Exception as e:
        raise CustomException(e,sys)