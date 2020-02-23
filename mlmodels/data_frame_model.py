from functools import wraps
import pandas as pd
import mlflow.pyfunc

from mlmodels.data_frame_schema import DataFrameSchema, _infer_data_frame_schema_from_df
from mlmodels.base_classes import BaseModel
from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


########################################################################################################
# Data frame model class
########################################################################################################
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

        self_var.feature_df_schema = _infer_data_frame_schema_from_df(X)

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

    @infer_target_df_schema_from_fit
    @infer_target_df_schema_from_fit
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
