import pickle as pickle
from datetime import datetime
from abc import ABCMeta, abstractmethod
from functools import wraps
import numpy as np
import pandas as pd
import mlflow.pyfunc
from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


########################################################################################################
# Basemodel classes
########################################################################################################
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


class DataFrameModel(BaseModel, metaclass=ABCMeta):
    """
    Base class for data frame models

    The class inherits from BaseModelClass, but has the fit and predict methods should work on pandas data frames and
    series. Features and dtypes should either be set on initialization or infered using the decorators
    infer_dataframe_dtypes_from_fit and infer_dataframe_features_from_fit.
    Having the dtypes allows us to validate the model input using the decorator validate_prediction_input
    and get the open api schema with get_open_api_yaml or get_open_api_dict.
    """
    def __init__(
            self,
            features=None,
            categorical_columns=None,
            feature_dtypes=None,
            target_dtype=None,
            possible_categorical_column_values=None,
    ):
        super().__init__()
        self.features = features
        self.categorical_columns = categorical_columns
        self.feature_dtypes = feature_dtypes
        self.target_dtype = target_dtype
        self.possible_categorical_column_values = possible_categorical_column_values

    ACCEPTED_DTYPES = (
        np.dtype('int64'),
        np.dtype('int32'),
        np.dtype('float64'),
        np.dtype('float32'),
        np.dtype('O'),
    )

    DTYPE_TO_JSON_TYPE_MAP = {
        np.dtype('int64'): {'type': 'number', 'format': 'integer'},
        np.dtype('int32'): {'type': 'number', 'format': 'integer'},
        np.dtype('float64'): {'type': 'number', 'format': 'float'},
        np.dtype('float32'): {'type': 'number', 'format': 'float'},
        np.dtype('O'): {'type': 'string'},
    }

    def get_model_input_record_field_schema(self):
        zipped_feature_dtype_pairs = zip(self.feature_dtypes.index, self.feature_dtypes)
        return {feature: self.DTYPE_TO_JSON_TYPE_MAP[dtype] for (feature, dtype) in zipped_feature_dtype_pairs}

    def get_open_api_yaml(self):
        return open_api_yaml_specification(
            model_input_record_field_schema_dict=self.get_model_input_record_field_schema(),
            possible_categorical_column_values=(self.possible_categorical_column_values or {}),
            model_target_field_schema_dict=self.DTYPE_TO_JSON_TYPE_MAP[self.target_dtype]
        )

    def get_open_api_dict(self):
        return open_api_dict_specification(
            model_input_record_field_schema_dict=self.get_model_input_record_field_schema(),
            possible_categorical_column_values=(self.possible_categorical_column_values or {}),
            model_target_field_schema_dict=self.DTYPE_TO_JSON_TYPE_MAP[self.target_dtype]
        )

    def convert_model_input_dtypes(self, model_input):
        """If types inferred py pandas to not match the required dtypes,
        we try to convert them."""
        dtype_dict = self.feature_dtypes.astype(str).to_dict()
        return model_input.astype(dtype_dict)

    def model_input_from_dict(self, dict_data):
        """Read data from record type deictionary representation"""

        model_input = pd.DataFrame.from_records(dict_data['data'])
        return self.convert_model_input_dtypes(model_input)


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def infer_dataframe_dtypes_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if isinstance(self_var, DataFrameModel) is False:
            raise ValueError(
                "The decorator only works on fit methods for objects of type DataFrameModel."
            )

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        if isinstance(y, pd.Series) is False:
            raise ValueError(
                "y must be a pandas Series."
            )

        if self_var.features is None:
            raise ValueError("features attribute must be set. It should be a list of features")

        if all(X[self_var.features].dtypes.isin(self_var.ACCEPTED_DTYPES)):
            self_var.feature_dtypes = X[self_var.features].dtypes
        else:
            raise ValueError(f"Dtypes of columns of X must be in {self_var.ACCEPTED_DTYPES}]")

        if y.dtypes in self_var.ACCEPTED_DTYPES:
            self_var.target_dtype = y.dtypes
        else:
            raise ValueError(f"Dtype of y must be in {self_var.ACCEPTED_DTYPES}]")

        return func(*args)

    return wrapper


def infer_category_feature_values_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if self_var.categorical_columns is None:
            return func(*args)
        else:
            if isinstance(self_var, DataFrameModel) is False:
                raise ValueError(
                    "The decorator only works on fit methods for objects of type DataFrameModel."
                )

            if isinstance(X, pd.DataFrame) is False:
                raise ValueError(
                    "X must be a pandas DataFrame."
                )

            if isinstance(y, pd.Series) is False:
                raise ValueError(
                    "y must be a pandas Series."
                )

            if self_var.features is None:
                raise ValueError("features attribute must be set. It should be a list of features")

            if not set(self_var.categorical_columns).issubset(set(self_var.features).union(set(y.name))):
                raise ValueError("categorical_features must be a subset of the union of features and target name")

            categorical_features = [cat_feat for cat_feat in self_var.categorical_columns if cat_feat in self_var.features]
            categorical_target = [cat_feat for cat_feat in self_var.categorical_columns if cat_feat == y.name]

            self_var.possible_categorical_column_values = {
                categorical_feature: list(X[categorical_feature].unique()) for categorical_feature in categorical_features
            }

            if categorical_target:
                self_var.possible_categorical_column_values[categorical_target] = list(X[categorical_target].unique())

            return func(*args)

    return wrapper


def infer_dataframe_features_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        if isinstance(self_var, DataFrameModel) is False:
            raise ValueError(
                "The decorator only works on fit methods for objects of type DataFrameModel."
            )
        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        self_var.features = list(X.columns)

        return func(*args)

    return wrapper


def validate_prediction_input(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        if isinstance(self_var, DataFrameModel) is False:
            raise ValueError(
                "The decorator only works on fit methods for objects of type DataFrameModel."
            )

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        if self_var.features is None:
            raise ValueError("features attribute must be set. It should be a list of features")

        if self_var.feature_dtypes is None:
            raise ValueError("dtypes attribute must be set. It should be a of the type pandas.DataFrame.dtypes")

        if not (set(X.columns) >= set(self_var.features)):
            raise ValueError(f"The following features must be in X: {self_var.features}")

        if not X[self_var.features].dtypes.to_dict() == self_var.feature_dtypes.to_dict():
            raise ValueError(f"Dtypes must be: {self_var.feature_dtypes.to_dict()}")

        if not all(X[self_var.features].dtypes.isin(self_var.ACCEPTED_DTYPES)):
            raise ValueError(f"Dtypes of columns of X must be in {self_var.ACCEPTED_DTYPES}]")

        return_values = func(*args)
        return return_values

    return wrapper


########################################################################################################
#
########################################################################################################
class FeatureSplitModel(DataFrameModel):
    MODEL_NAME = 'Feature split meta model'

    def __init__(
            self,
            features=None,
            categorical_columns=None,
            feature_dtypes=None,
            target_dtype=None,
            possible_categorical_column_values=None,
            group_column=None,
            group_model_dict=None
    ):
        super().__init__()
        self.features = features
        self.categorical_columns = categorical_columns
        self.feature_dtypes = feature_dtypes
        self.target_dtype = target_dtype
        self.possible_categorical_column_values = possible_categorical_column_values
        self.group_model_dict = group_model_dict
        self.group_column = group_column

    @infer_dataframe_dtypes_from_fit
    @infer_category_feature_values_from_fit
    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"

        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            self.group_model_dict[group].fit(X[mask], y[mask])

    @validate_prediction_input
    def predict(self, X):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
        X = X.copy()
        X['prediction'] = float('NaN')
        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            X.loc[mask, 'prediction'] = self.group_model_dict[group].predict(X[mask])

        return X['prediction'].values


########################################################################################################
# Wrapper for mlflow
########################################################################################################
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


########################################################################################################
# Base transformer classes
########################################################################################################
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