from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from mlmodels import (
    BaseModel,
    data_frame_model
)


@data_frame_model
class RandomForestRegressorModel(BaseModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features,
            categorical_columns=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.categorical_columns = categorical_columns
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X):
        predictions = self.model.predict(X[self.features])
        return predictions


@data_frame_model
class RandomForestClassifierModel(BaseModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features,
            categorical_columns=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.categorical_columns = categorical_columns
        self.random_forest_params = random_forest_params
        self.model = RandomForestClassifier(**random_forest_params)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_name = y.name
        return self

    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_series = pd.Series(data=predictions_array, name=self.target_name)
        return predictions_series
