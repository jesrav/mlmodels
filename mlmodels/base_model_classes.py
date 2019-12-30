import pickle as pickle
from datetime import datetime
from abc import ABCMeta, abstractmethod
from functools import wraps
import numpy as np
import pandas as pd
from marshmallow_dataframe import RecordsDataFrameSchema
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
import mlflow.pyfunc
from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


class BaseModel(metaclass=ABCMeta):
    """
    Base class for models

    The class has a save and load method for serializing model objects.
    It enforces implementation of a fit and predict method and a model name attribute.
    """
    def __init__(self):
        self.model_initiated_dt = datetime.utcnow()

    @property
    @classmethod
    @abstractmethod
    def MODEL_NAME(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def __str__(self):
        return f'Model: {self.MODEL_NAME},  initiated at {self.model_initiated_dt}'

    def save(self, **kwargs):
        """Serialize model to file or variable
        """
        serialize_dict = self.__dict__

        if "fname" in kwargs.keys():
            fname = kwargs["fname"]
            with open(fname, "wb") as f:
                pickle.dump(serialize_dict, f)
        else:
            pickled = pickle.dumps(serialize_dict)
            return pickled

    def load(self, serialized):
        """Deserialize model from file or variable"""
        assert isinstance(serialized, str) or isinstance(
            serialized, bytes
        ), "serialized must be a string (filepath) or a bytes object with the serialized model"
        model = self.__class__()

        if isinstance(serialized, str):
            with open(serialized, "rb") as f:
                serialize_dict = pickle.load(f)
        else:
            serialize_dict = pickle.loads(serialized)

        # Set attributes of model
        model.__dict__ = serialize_dict

        return model


def infer_dataframe_dtypes_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        assert isinstance(X, pd.DataFrame), 'X must be a DataFrame'
        if self_var.features is None:
            raise ValueError("features attribute must be set. It should be a list of features")

        if self_var.feature_dtypes is None:
            if all(X[self_var.features].dtypes.isin(self_var.ACCEPTED_DTYPES)):
                self_var.feature_dtypes = X[self_var.features].dtypes
            else:
                raise ValueError(f"Dtypes of columns of X must be in {self_var.ACCEPTED_DTYPES}]")

        if self_var.target_dtype is None:
            if y.dtypes in self_var.ACCEPTED_DTYPES:
                self_var.target_dtype = y.dtypes
            else:
                raise ValueError(f"Dtype of y must be in {self_var.ACCEPTED_DTYPES}]")

        func(*args)

    return wrapper

def infer_dataframe_features_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        assert isinstance(X, pd.DataFrame), 'X must be a DataFrame'
        if self_var.features is None:
            self_var.features = list(X.columns)

        func(*args)

    return wrapper

class DataFrameModel(BaseModel, metaclass=ABCMeta):

    def __init__(
            self,
            features=None,
            feature_dtypes=None,
            target_dtype=None,
    ):
        super().__init__()
        self.model_initiated_dt = datetime.utcnow()
        self.features = features
        self.feature_dtypes = feature_dtypes
        self.target_dtype = target_dtype

    @property
    @classmethod
    @abstractmethod
    def MODEL_NAME(self):
        pass

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

    def get_model_input_schema(self):
        class ModelInputSchema(RecordsDataFrameSchema):
            """Automatically generated schema for model input dataframe"""
            class Meta:
                dtypes = self.feature_dtypes
        return ModelInputSchema

    def record_dict_to_model_input(self, dict_data):
        model_input_schema = self.get_model_input_schema()()
        return model_input_schema.load(dict_data)

    def get_record_field_schema(self):
        spec = APISpec(
            title="Prediction open api spec",
            version="1.0.0",
            openapi_version="3.0.2",
            plugins=[MarshmallowPlugin()],
        )
        model_input_schema_class = self.get_model_input_schema()
        spec.components.schema("predict", schema=model_input_schema_class)
        spec_dict = spec.to_dict()
        record_field_schema = spec_dict['components']['schemas']['Record']['properties']
        return record_field_schema

    def get_open_api_yaml(self):
        record_field_schema = self.get_record_field_schema()
        return open_api_yaml_specification(
            feature_dict=record_field_schema,
            target_dict=self.TARGET_TO_JSON_TYPE_MAP[self.target_dtype]
        )

    def get_open_api_dict(self):
        record_field_schema = self.get_record_field_schema()
        return open_api_dict_specification(
            feature_dict=record_field_schema,
            target_dict=self.TARGET_TO_JSON_TYPE_MAP[self.target_dtype]
        )


class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


class BaseTransformer(metaclass=ABCMeta):
    """
    Base class for Transformera

    The class has a save and load method for serializing model objects.
    It enforces implementation of a fit and transform method and a model name attribute.
    """
    def __init__(self):
        self.transformer_initiated_dt = datetime.utcnow()

    @property
    @classmethod
    @abstractmethod
    def TRANSFORMER_NAME(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    def __str__(self):
        return f'Transformer: {self.TRANSFORMER_NAME},  initiated at {self.transformer_initiated_dt}'

    def save(self, **kwargs):
        """Serialize model to file or variable
        """
        serialize_dict = self.__dict__

        if "fname" in kwargs.keys():
            fname = kwargs["fname"]
            with open(fname, "wb") as f:
                pickle.dump(serialize_dict, f)
        else:
            pickled = pickle.dumps(serialize_dict)
            return pickled

    def load(self, serialized):
        """Deserialize transformer from file or variable
        """
        assert isinstance(serialized, str) or isinstance(
            serialized, bytes
        ), "serialized must be a string (filepath) or a bytes object with the serialized model"
        model = self.__class__()

        if isinstance(serialized, str):
            with open(serialized, "rb") as f:
                serialize_dict = pickle.load(f)
        else:
            serialize_dict = pickle.loads(serialized)

        # Set attributes of model
        model.__dict__ = serialize_dict

        return model