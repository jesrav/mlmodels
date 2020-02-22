from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from mlmodels import (
    BaseModel,
    DataFrameModel,
    infer_category_values_from_fit,
    infer_feature_dtypes_from_fit,
    infer_target_dtypes_from_fit,
    validate_prediction_input_and_output,
)


class RandomForestRegressorModel(BaseModel, DataFrameModel):
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

    @infer_category_values_from_fit
    @infer_feature_dtypes_from_fit
    @infer_target_dtypes_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    @validate_prediction_input_and_output
    def predict(self, X):
        predictions = self.model.predict(X[self.features])
        return predictions


class RandomForestClassifierModel(BaseModel, DataFrameModel):
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

    @infer_category_values_from_fit
    @infer_feature_dtypes_from_fit
    @infer_target_dtypes_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_name = y.name
        return self

    @validate_prediction_input_and_output
    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_series = pd.Series(data=predictions_array, name=self.target_name)
        return predictions_series
