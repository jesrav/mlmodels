from functools import wraps
from typing import List, Union
import pandas as pd
import mlflow.pyfunc

from mlmodels.data_frame_schema import (
    DataFrameSchema,
    _infer_data_frame_schema_from_df,
    _get_enums_from_data_frame,
)
from mlmodels.base_classes import BaseModel
from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


########################################################################################################
# Data frame model mixin class
########################################################################################################
class DataFrameModelMixin:
    """Data frame model class

    The data frame model mixin class can be used to add functionality to a model class that takes a
    Pandas DataFrame as input and produces predictions in the form of a Pandas DataFrame.
    """

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

        self_var.feature_df_schema = _infer_data_frame_schema_from_df(X)

        if hasattr(self_var, 'feature_enum_columns'):
            if self_var.feature_enum_columns:
                enum_dict = _get_enums_from_data_frame(X, self_var.feature_enum_columns)
                for feature_column, enum in enum_dict.items():
                    self_var.feature_df_schema.modify_column(feature_column, enum=enum)

        return func(*args)

    return wrapper


def infer_target_df_schema_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if isinstance(y, pd.DataFrame) is False:
            raise ValueError(
                "y must be a pandas DataFrame."
            )

        self_var.target_df_schema = _infer_data_frame_schema_from_df(y)

        if hasattr(self_var, 'target_enum_columns'):
            if self_var.target_enum_columns:
                enum_dict = _get_enums_from_data_frame(y, self_var.target_enum_columns)
                for target_column, enum in enum_dict.items():
                    self_var.target_df_schema.modify_column(target_column, enum=enum)

        return func(*args)
    return wrapper


def validate_prediction_input_and_output(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        _ = self_var.feature_df_schema.validate_df(X)

        return_values = func(*args)

        _ = self_var.target_df_schema.validate_df(return_values)

        return return_values

    return wrapper


########################################################################################################
# Data frame model that uses separate model for each category of a feature.
########################################################################################################
class FeatureSplitModel(BaseModel, DataFrameModelMixin):
    MODEL_NAME = 'Feature split meta model'

    def __init__(
            self,
            features=None,
            feature_enum_columns=None,
            target_enum_columns=None,
            group_column=None,
            group_model_dict=None
    ):
        super().__init__()
        self.features = features
        self.target_columns = None,
        self.feature_enum_columns = feature_enum_columns,
        self.target_enum_columns = target_enum_columns,
        self.group_model_dict = group_model_dict
        self.group_column = group_column

    @infer_target_df_schema_from_fit
    @infer_target_df_schema_from_fit
    def fit(self, X, y):
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"

        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            self.group_model_dict[group].fit(X[mask], y[mask])

        self.target_columns = y.columns

    @validate_prediction_input_and_output
    def predict(self, X):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
        X = X.copy()
        X['prediction'] = float('NaN')
        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            X.loc[mask, 'prediction'] = self.group_model_dict[group].predict(X[mask])
            prediction_df = pd.DataFrame(data=X['prediction'].values, columns=self.target_columns)
        return prediction_df


# ########################################################################################################
# # Wrapper for mlflow
# ########################################################################################################
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
