from sklearn.ensemble import RandomForestRegressor
from mlmodels import DataFrameModel, infer_dataframe_dtypes_from_fit, validate_prediction_input


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

    @infer_dataframe_dtypes_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self.model

    @validate_prediction_input
    def predict(self, X):
        predictions = self.model.predict(X[self.features])
        return predictions
