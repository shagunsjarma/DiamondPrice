import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

def save_obj(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.value())[i]
            model.train(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(model.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.info("Error occured at model training")
        CustomException(e, sys)