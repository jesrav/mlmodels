import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlmodels import (
    MLFlowWrapper,
    ModelMethodColumnInfo,
    Interval,
    Column, DataFrameSchema)
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
    data['group1'] = data['group1'].astype('int64')
    data['group2'] = data['group2'].astype('int64')

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Fit model, make predictions and evaluate
    model = RandomForestClassifierModel(
        features=train_x.columns,
        random_forest_params={'n_estimators': 100, 'max_depth': 15},
    )

    # Set info about what column info to infer on fit
    for method_name in ['predict', 'predict_proba']:
        model.set_model_method_column_info(
            ModelMethodColumnInfo(
                method_name,
                input_enum_columns=['group1', 'group2'],
                output_enum_columns=['quality'],
                input_interval_columns=['chlorides', 'free sulfur dioxide'],
                input_interval_percent_buffer=25,
            )
        )

    model.fit(train_x, train_y)

    def set_predict_proba_method_output_schema_from_fitted_model(model):

        probability_column_names = [
            f'probability of quality = {class_}' for class_ in model.model.classes_
        ]

        columns = [
            Column(
                name,
                dtype='float64',
                interval=Interval(0, 1)
            ) for name in probability_column_names
        ]

        model.set_model_method_output_schema(
            'predict_proba',
            DataFrameSchema(columns)
        )

        set_predict_proba_method_output_schema_from_fitted_model(model)

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