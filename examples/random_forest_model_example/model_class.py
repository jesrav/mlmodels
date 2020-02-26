from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
from mlmodels import (
    BaseModel,
    DataFrameModelMixin,
    infer_feature_df_schema_from_fit,
    infer_target_df_schema_from_fit,
    validate_prediction_input_and_output,
)


class RandomForestRegressorModel(BaseModel, DataFrameModelMixin):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.target_columns = None
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    @infer_feature_df_schema_from_fit
    @infer_target_df_schema_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    @validate_prediction_input_and_output
    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df


class RandomForestClassifierModel(BaseModel, DataFrameModelMixin):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features,
            feature_enum_columns=None,
            target_enum_columns=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.target_columns = None,
        self.feature_enum_columns = feature_enum_columns
        self.target_enum_columns = target_enum_columns
        self.random_forest_params = random_forest_params
        self.model = RandomForestClassifier(**random_forest_params)

    @infer_feature_df_schema_from_fit
    @infer_target_df_schema_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    @validate_prediction_input_and_output
    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df
