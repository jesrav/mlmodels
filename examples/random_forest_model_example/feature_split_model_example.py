from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_boston
from mlmodels import FeatureSplitModel
from model_class import RandomForestRegressorModel
from mlmodels import MLFlowWrapper
import mlflow.pyfunc


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))

    code_path = str(dir_path / Path('model_class.py'))
    model_path = str(dir_path / Path('model_output/feature_split_model'))
    conda_env_path = str(dir_path / Path('conda.yaml'))

    # Load Boston housing data
    boston = load_boston()
    x = pd.DataFrame(boston.data)
    x.columns = boston.feature_names
    x['group'] = np.random.choice(['group1', 'group2'], len(x))
    y = pd.DataFrame(boston.target)
    y.columns = ['price']

    # Split the data into training and test sets. (0.75, 0.25) split.
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Create dictionary of individual models. In this cas all the same.
    features_individual_models = [
        "AGE",
        "B",
        "CHAS",
        "CRIM",
    ]

    group_model_dict = {group: RandomForestRegressorModel(
        features=features_individual_models,
    ) for group in x['group'].unique()}

    # Create feature split model
    features = features_individual_models + ["group"]
    model = FeatureSplitModel(
        features=features,
        group_column="group",
        group_model_dict=group_model_dict
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
