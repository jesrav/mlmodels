import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlmodels import MLFlowWrapper
from model_class import RandomForestRegressorModel
import json

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    load_dotenv(find_dotenv())

    # Read the wine-quality csv file from the URL
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Fit model, make predictions and evaluate
    model = RandomForestRegressorModel(
        features=train_x.columns,
        random_forest_params={'n_estimators': 100, 'max_depth': 15},
    )
    model.fit(train_x, train_y)

    predicted_qualities = model.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    from pprint import pprint
    with open('test.yaml', 'w') as f:
        f.write(model.get_open_api_yaml())
    pprint(model.get_open_api_yaml())
    print(model.feature_dtypes)
    print(model.target_dtype)
