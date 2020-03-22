import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_boston
from mlmodels import MLFlowWrapper, SKLearnWrapper, infer_from_fit
import mlflow.pyfunc

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model_output/sklearn_wrapper_model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    # Read the wine-quality csv file from the URL
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Fit model, make predictions and evaluate
    model = SKLearnWrapper(
        features=[
            "alcohol",
            "chlorides",
            "citric acid",
            "density",
            "fixed acidity",
            "free sulfur dioxide"
        ],
        sklearn_model_class=LogisticRegression,
        sklearn_model_params={'C': 1},
    )

    # decorate model to infer schemas from fit
    model = infer_from_fit(
        feature_df_schema=True,
        target_df_schema=True,
        methods_with_features_as_input=['predict'],
        validate_input_output_method_list=['predict']
    )(model)

    print(type(model))

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
