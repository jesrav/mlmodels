from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from mlmodels import (
    BaseModel,
    DataFrameModelMixin,
    infer_from_fit,
    infer_feature_df_schema_from_fit,
    infer_target_df_schema_from_fit,
    validate_method_input_and_output,
)


class RandomForestRegressorModel(BaseModel, DataFrameModelMixin):
    MODEL_NAME = 'Random forest regression model'

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

    @infer_feature_df_schema_from_fit(method_list=['predict'])
    @infer_target_df_schema_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    @validate_method_input_and_output
    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df


@infer_from_fit(
    feature_df_schema=True,
    target_df_schema=True,
    methods_with_features_as_input=['predict', 'predict_proba'],
    validate_input_output_method_list=['predict', 'predict_proba']
)
class RandomForestClassifierModel(BaseModel, DataFrameModelMixin):

    def __init__(
            self,
            features,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.target_columns = None,
        self.random_forest_params = random_forest_params
        self.model = RandomForestClassifier(**random_forest_params)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df

    def predict_proba(self, X):
        probability_array = self.model.predict_proba(X[self.features])
        probability_column_names = [f'probability of quality = {class_}' for class_ in self.model.classes_]
        probability_df = pd.DataFrame(data=probability_array, columns=probability_column_names)
        return probability_df


class FeatureSplitModel(BaseModel, DataFrameModelMixin):

    def __init__(
            self,
            features,
            group_column,
            group_model_dict,
    ):
        super().__init__()
        self.features = features
        self.group_column = group_column
        self.group_model_dict = group_model_dict
        self.target_columns = None

    @infer_target_df_schema_from_fit
    @infer_feature_df_schema_from_fit(method_list=['predict'])
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"

        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            self.group_model_dict[group].fit(X[mask], y[mask])

        self.target_columns = y.columns

    @validate_method_input_and_output
    def predict(self, X):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
        X = X.copy()

        X.append(pd.Series(name='prediction'))
        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            X.loc[mask, 'prediction'] = self.group_model_dict[group].predict(X[mask])
            prediction_df = pd.DataFrame(data=X['prediction'], columns=self.target_columns)
        return prediction_df


