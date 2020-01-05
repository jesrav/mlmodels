import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlmodels import FeatureSplitModel
from model_class import RandomForestRegressorModel


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return rmse, mae


if __name__ == '__main__':

    # Read the wine-quality csv file from the URL
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    data = pd.read_csv(csv_url, sep=';')

    # Create 3 randomly assigned groups
    data['group'] = np.random.choice(3, len(data))

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Create dictionary of individual models. In this cas all the same.
    features_individual_models = ["density", "chlorides", "alcohol"]
    group_model_dict = {group: RandomForestRegressorModel(
        features=features_individual_models,
        random_forest_params={'n_estimators': 100, 'max_depth': 15}
    ) for group in data['group'].unique()}

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
