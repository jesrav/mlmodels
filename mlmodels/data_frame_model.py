from typing import List, Dict
from functools import wraps
import numpy as np
import pandas as pd
import mlflow.pyfunc
import pandera as pa

from mlmodels.base_classes import BaseModel
from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


########################################################################################################
# Data frame model class
########################################################################################################
_ACCEPTED_DTYPES = (
    'int64',
    'int32',
    'float64',
    'float32',
    'object',
    'string',
)

_dtype_to_pandera_map = {
    'int64': pa.Int,
    'int32': pa.Int,
    'float64': pa.Float,
    'float32': pa.Float,
    'object': pa.String,
    'string': pa.String,
}


def _validate_name(name):
    if not isinstance(name, str):
        raise TypeError('name must be string.')


def _validate_dtype(dtype):
    if not isinstance(dtype, str):
        raise TypeError('dtype must be string.')
    if not dtype in _ACCEPTED_DTYPES:
        raise ValueError(f'dtype must be one of the accepted dtypes: {_ACCEPTED_DTYPES}')


def _validate_enum(enum):
    if not isinstance(enum, list):
        raise TypeError('enum must be list.')


def _validate_column_input(name, dtype, enum):
    _validate_name(name)
    _validate_dtype(dtype)
    _validate_enum(enum)


class Column:

    def __init__(self, name: str, dtype: str, enum:List = []):

        _validate_column_input(name, dtype, enum)

        self.name = name
        self.dtype = dtype
        self.enum = enum

    def update_enum(self, enum):
        _validate_enum(enum)
        self.enum = enum

    def __repr__(self):
        return f'Column{{name: {self.name}, dtype: {self.dtype}, enum: {self.enum}}}'


def _pandera_data_frame_schema_from_columns(columns:List):

    data_frame_schema_dict = {}
    for col in columns:
        if col.enum:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype],
                pa.Check.isin(col.enum)
            )
        else:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype]
            )
    return pa.DataFrameSchema(data_frame_schema_dict)


class DataFrameSchema:

    def __init__(self, columns: list):
        self.columns = columns
        self._data_frame_schema = _pandera_data_frame_schema_from_columns(columns)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate pandas data frame

        Parameters
        ----------
        df: pandas DataFrame

        Raises:
        -------
        SchemaError
            If df does not conform to the schema, a schema error is raised.

        Returns
        -------
        DataFrame
            Validated data frame
        """
        return self._data_frame_schema.validate(df)

    def __repr__(self):
        return f'DataFrameSchema{{columns: {self.columns}}}'


df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})


def _infer_data_frame_schema(df: pd.DataFrame) -> DataFrameSchema:
    dtype_dict = df.dtypes.astype(str).to_dict()
    return DataFrameSchema([Column(k, dtype_dict[k]) for k in dtype_dict])


class DataFrameModel:
    """Data frame model class

    The data frame model class can be used to add functionality to a model class that takes a
    Pandas DataFrame as input and produces predictions in the form of a Pandas Series or DataFrame.
    """
    def __init__(self):
        self.feature_df_schema=None
        self.target_df_schema=None

    def set_feature_df_schema(self, feature_df_schema:DataFrameSchema):
        self.feature_df_schema = feature_df_schema

    def set_target_df_schema(self, target_df_schema:DataFrameSchema):
        self.target_df_schema = target_df_schema

    def get_open_api_yaml(self):
        """Get the open API spec for the the model predictions in a YAML representation.

        Returns
        -------
        str
           YAML representation of the open API spec for the the model predictions
        """
        return open_api_yaml_specification(
            feature_df_schema=self.feature_df_schema,
            target_df_schema=self.target_df_schema
        )

    def get_open_api_dict(self):
        """Get the open API spec for the the model predictions in a dictionary representation.

        Returns
        -------
        dict
           Dictionary representation of the open API spec for the the model predictions
        """
        return open_api_dict_specification(
            feature_df_schema=self.feature_df_schema,
            target_df_schema=self.target_df_schema
        )


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def infer_feature_df_schema_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        self_var.feature_df_schema = _infer_data_frame_schema(X)

        return func(*args)

    return wrapper


def infer_target_dtypes_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if isinstance(y, pd.DataFrame) is False:
            raise ValueError(
                "y must be a pandas DataFrame."
            )

        self_var.target_df_schema = _infer_data_frame_schema(y)

        return func(*args)

    return wrapper


# def infer_category_values_from_fit(func):
#
#     @wraps(func)
#     def wrapper(*args):
#         self_var = args[0]
#         X = args[1]
#         y = args[2]
#
#         if self_var.categorical_columns is None:
#             return func(*args)
#         else:
#
#             if isinstance(X, pd.DataFrame) is False:
#                 raise ValueError(
#                     "X must be a pandas DataFrame."
#                 )
#
#             if (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)) is False:
#                 raise ValueError(
#                     "y must be a pandas Series or DataFrame."
#                 )
#
#             if self_var.features is None:
#                 raise ValueError("features attribute must be set. It should be a list of features")
#
#             if not set(self_var.categorical_columns).issubset(set(self_var.features).union({y.name})):
#                 raise ValueError("categorical_features must be a subset of the union of features and target name")
#
#             categorical_features = [
#                 cat_feat for cat_feat in self_var.categorical_columns
#                 if cat_feat in self_var.features
#             ]
#             categorical_target = [
#                 cat_feat for cat_feat in self_var.categorical_columns
#                 if cat_feat == y.name
#             ][0]
#
#             self_var.possible_categorical_column_values = {
#                 categorical_feature: list(X[categorical_feature].unique())
#                 for categorical_feature in categorical_features
#             }
#
#             if categorical_target:
#                 self_var.possible_categorical_column_values[categorical_target] = list(y.unique())
#
#             return func(*args)
#
#     return wrapper


def validate_prediction_input_and_output(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        _ = self_var.feature_df_schema.validate(X)

        return_values = func(*args)

        _ = self_var.target_df_schema.validate(return_values)

        return return_values

    return wrapper


########################################################################################################
# Data frame model that uses separate model for each category of a feature.
########################################################################################################
class FeatureSplitModel(BaseModel, DataFrameModel):
    MODEL_NAME = 'Feature split meta model'

    def __init__(
            self,
            features=None,
            categorical_columns=None,
            group_column=None,
            group_model_dict=None
    ):
        super().__init__()
        self.features = features
        self.categorical_columns = categorical_columns
        self.group_model_dict = group_model_dict
        self.group_column = group_column

    @infer_category_values_from_fit
    @infer_target_dtypes_from_fit
    @infer_target_dtypes_from_fit
    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"

        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            self.group_model_dict[group].fit(X[mask], y[mask])

    @validate_prediction_input_and_output
    def predict(self, X):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
        X = X.copy()
        X['prediction'] = float('NaN')
        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            X.loc[mask, 'prediction'] = self.group_model_dict[group].predict(X[mask])

        return X['prediction'].values


# ########################################################################################################
# # Wrapper for mlflow
# ########################################################################################################
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
