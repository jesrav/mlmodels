import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from marshmallow_dataframe import RecordsDataFrameSchema
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin

from mlmodels import BaseModel
from mlmodels.openapi_yaml_template import open_api_yaml_specification


class RandomForestRegressorModel(BaseModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features=None,
            feature_dtypes=None,
            target_dtype=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30}
    ):
        super().__init__()
        self.features = features
        self.feature_dtypes = feature_dtypes
        self.target_dtype = target_dtype
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    ACCEPTED_DTYPES = (
        np.dtype('int64'),
        np.dtype('int32'),
        np.dtype('float64'),
        np.dtype('float32'),
        np.dtype('O'),
    )

    TARGET_TO_JSON_TYPE_MAP = {
        np.dtype('int64'): {'type': 'number', 'format': 'integer'},
        np.dtype('int32'): {'type': 'number', 'format': 'integer'},
        np.dtype('float64'): {'type': 'number', 'format': 'float'},
        np.dtype('float32'): {'type': 'number', 'format': 'float'},
        np.dtype('O'): {'type': 'string'},
    }

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), 'X must be a DataFrame'
        assert isinstance(y, pd.Series), 'y must be a DataFrame'
        if self.feature_dtypes is None:
            if all(X[self.features].dtypes.isin(self.ACCEPTED_DTYPES)):
                self.feature_dtypes = X[self.features].dtypes
            else:
                raise ValueError("Dtypes of columns of X must be in {self.ACCEPTED_DTYPES}]")
        if self.target_dtype is None:
            if y.dtypes in self.ACCEPTED_DTYPES:
                self.target_dtype = y.dtypes
            else:
                raise ValueError("Dtype of y must be in {self.ACCEPTED_DTYPES}]")

        self.model.fit(X[self.features], y)

        return self.model

    def predict(self, X):
        assert isinstance(X, pd.DataFrame), 'X must be a DataFrame'
        # assert all(list(X.columns) == self.features), f'The following features must be in X: {self.features}'
        assert X[self.features].dtypes.to_dict() == self.feature_dtypes.to_dict(), f'Dtypes must be: {self.feature_dtypes.to_dict()}'
        predictions = self.model.predict(X[self.features])
        return predictions

    def get_model_input_schema(self):
        class ModelInputSchema(RecordsDataFrameSchema):
            """Automatically generated schema for model input dataframe"""

            class Meta:
                dtypes = self.feature_dtypes

        return ModelInputSchema

    def record_dict_to_model_input(self, dict_data):
        model_input_schema = self.get_model_input_schema()()
        return model_input_schema.load(dict_data)

    def get_open_api_yaml(self):
        # Create an APISpec
        spec = APISpec(
            title="Prediction open api spec",
            version="1.0.0",
            openapi_version="3.0.2",
            plugins=[MarshmallowPlugin()],
        )
        ModelInputSchema = self.get_model_input_schema()
        spec.components.schema("predict", schema=ModelInputSchema)
        spec_dict = spec.to_dict()
        record_field_specs = spec_dict['components']['schemas']['Record']['properties']
        return open_api_yaml_specification(
            feature_dict=record_field_specs,
            target_dict = self.TARGET_TO_JSON_TYPE_MAP[self.target_dtype]
        )