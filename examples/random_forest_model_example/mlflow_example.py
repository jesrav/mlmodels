import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlmodels import MLFlowWrapper
from model_class import RandomForestRegressorModel
import mlflow


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    load_dotenv(find_dotenv())

    if os.environ.get("MLFLOW_SERVER_URL"):
        mlflow.set_tracking_uri(
           'http://{user}:{psw}@{server}'.format(
               server=os.environ.get("MLFLOW_SERVER_URL"),
               user=os.environ.get("MLFLOW_USER"),
               psw=os.environ.get("MLFLOW_PASSWORD"),
           )
        )

    mlflow.set_experiment('test_experiment')

    # Read the wine-quality csv file from the URL
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality", "pH"], axis=1)
    test_x = test.drop(["quality", "pH"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    with mlflow.start_run():
        # Fit model, make predictions and evaluate
        model = RandomForestRegressorModel(
            features=train_x.columns,
            random_forest_params={'n_estimators': 100, 'max_depth': 15},
        )
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)

        (rmse, mae) = eval_metrics(test_y, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)

        mlflow.log_param("model_params", model.random_forest_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        model_mlflow = MLFlowWrapper(model)

        # log model
        mlflow.pyfunc.log_model(model_path, code_path=[code_path], python_model=model_mlflow)

        # mlflow.pyfunc.save_model(
        #     path=str(model_path),
        #     python_model=model_mlflow,
        #     code_path=[code_path])
