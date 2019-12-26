from mlmodels import BaseModel
from sklearn.ensemble import RandomForestRegressor
from marshmallow_dataframe import RecordsDataFrameSchema
import json

class RandomForestRegressorModel(BaseModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features,
            input_dtypes=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30}
    ):
        super().__init__()
        self.features = features
        self.input_dtypes = None
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    def fit(self, X, y):
        if self.input_dtypes is None:
            self.input_dtypes = X[self.features].dtypes
        self.model.fit(X[self.features], y)
        return self.model

    def predict(self, X):
        assert all(list(X.columns) == self.features), f'The following features must be in X: {self.features}'
        assert X[self.features].dtypes.to_dict() == self.input_dtypes.to_dict(), f'Dtypes must be: {self.input_dtypes.to_dict()}'
        predictions = self.model.predict(X[self.features])
        return predictions

    def get_model_input_schema(self):
        class ModelInputSchema(RecordsDataFrameSchema):
            """Automatically generated schema for model input dataframe"""
            class Meta:
                dtypes = self.input_dtypes
        return ModelInputSchema()

    def json_to_model_input(self, json_data):
        json_data = json.loads(json_data)
        model_input_schema = self.get_model_input_schema()
        return model_input_schema.load(json_data)
