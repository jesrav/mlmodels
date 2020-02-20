import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlmodels import MLFlowWrapper
from model_class import RandomForestClassifierModel
import mlflow.pyfunc


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model_output/wine_model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    # Read the wine-quality csv file from the URL
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')

    # Create 3 randomly assigned groups
    data['group1'] = np.random.choice(3, len(data))
    data['group2'] = np.random.choice([3, 7], len(data))

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Fit model, make predictions and evaluate
    model = RandomForestClassifierModel(
        features=train_x.columns,
        categorical_columns=['group1', 'group2', 'quality'],
        random_forest_params={'n_estimators': 100, 'max_depth': 15},
    )
    model.fit(train_x, train_y)

    predicted_qualities = model.predict(test_x)

    print(predicted_qualities.head())
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