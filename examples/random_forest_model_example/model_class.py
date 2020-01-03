import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlmodels import DataFrameModel, infer_dataframe_dtypes_from_fit, infer_dataframe_features_from_fit, validate_prediction_input


class RandomForestRegressorModel(DataFrameModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30}
    ):
        super().__init__()
        self.features = features
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    @infer_dataframe_features_from_fit
    @infer_dataframe_dtypes_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self.model

    @validate_prediction_input
    def predict(self, X):
        # assert isinstance(X, pd.DataFrame), 'X must be a DataFrame'
        # assert sum([col in X.columns for col in self.features]) == len(self.features), f'The following features must be in X: {self.features}'
        # assert X[self.features].dtypes.to_dict() == self.feature_dtypes.to_dict(), f'Dtypes must be: {self.feature_dtypes.to_dict()}'
        predictions = self.model.predict(X[self.features])
        return predictions
