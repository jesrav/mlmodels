from sklearn.ensemble import RandomForestRegressor
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
