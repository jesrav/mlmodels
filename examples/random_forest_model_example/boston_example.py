import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_boston
from mlmodels import MLFlowWrapper
from model_class import RandomForestRegressorModel
import mlflow.pyfunc


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model_output/boston_model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    # Load Boston housing data
    boston = load_boston()
    x = pd.DataFrame(boston.data)
    x.columns = boston.feature_names
    y = pd.DataFrame(boston.target)
    y.columns = ['price']

    # Split the data into training and test sets. (0.75, 0.25) split.
    train_x, test_x, train_y, test_y = train_test_split(x, y)

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

    model_mlflow = MLFlowWrapper(model)

    mlflow.pyfunc.save_model(
        path=model_path,
        python_model=model_mlflow,
        code_path=[code_path],
        conda_env=conda_env_path
    )
