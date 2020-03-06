from functools import wraps
from typing import Union, List

import mlflow.pyfunc
import pandas as pd

from mlmodels.data_frame_schema import (
    DataFrameSchema,
    get_data_frame_schema_from_df)


class ModelMethodSchema:

    def __init__(
            self,
            method_name: str,
            input_schema: Union[None, DataFrameSchema] = None,
            output_schema: Union[None, DataFrameSchema] = None,
    ):
        self.method_name = method_name
        self.input_schema = input_schema
        self.output_schema = output_schema

    def set_input_schema(self, input_schema: DataFrameSchema):
        if not isinstance(input_schema, DataFrameSchema):
            raise TypeError('input_schema most be a DataFrameSchema instance.')
        self.input_schema = input_schema

    def set_output_schema(self, output_schema: DataFrameSchema):
        if not isinstance(output_schema, DataFrameSchema):
            raise TypeError('output_schema most be a DataFrameSchema instance.')
        self.output_schema = output_schema

    def __repr__(self):
        return (
            f'ModelMethodSchema{{method_name: {self.method_name},\n'
            f'input_schema: {self.input_schema},\n'
            f'output_schema: {self.output_schema}}}'
        )


class ModelMethodColumnInfo:
    def __init__(
            self,
            method_name: str,
            input_enum_columns: List[str] = None,
            output_enum_columns: List[str] = None,
            input_interval_columns: List[str] = None,
            input_interval_percent_buffer: float = 0.0

    ):

        self.method_name = method_name
        self.input_enum_columns = input_enum_columns
        self.output_enum_columns = output_enum_columns
        self.input_interval_columns = input_interval_columns
        self.input_interval_percent_buffer = input_interval_percent_buffer

    def __repr__(self):
        return (
            f'ModelMethodColumnInfo{{method_name: {self.method_name},\n'
            f'input_enum_columns: {self.input_enum_columns},\n'
            f'output_enum_columns: {self.output_enum_columns},\n'
            f'input_interval_columns: {self.input_interval_columns},\n'
            f'input_interval_percent_buffer: {self.input_interval_percent_buffer}}}'
        )


########################################################################################################
# Data frame model mixin class
########################################################################################################
class DataFrameModel:
    """Data frame model class
    """

    def __init__(self):
        self.model_method_column_info_dict = {}
        self.model_method_schema_dict = {}

    def set_model_method_column_info(self, model_method_column_info: ModelMethodColumnInfo):

        # _validate_model_method_column_info(model_method_column_info)

        if model_method_column_info.method_name in self.model_method_column_info_dict:
            raise ValueError(f'{model_method_column_info.method_name} has already been set.')
        self_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name))]

        if model_method_column_info.method_name not in self_methods:
            raise ValueError(
                f'{model_method_column_info.method_name} is not a method in {self}.'
            )

        self.model_method_column_info_dict[model_method_column_info.method_name] = model_method_column_info

    def set_model_method_schema(self, model_method_schema: ModelMethodSchema):

        # _validate_model_method_schema(model_method_schema)

        if model_method_schema.method_name in self.model_method_schema_dict:
            raise ValueError(f'{model_method_schema.method_name} has already been set.')

        self_methods = [method_name for method_name in dir(self)
                          if callable(getattr(self, method_name))]
        if model_method_schema.method_name not in self_methods:
            raise ValueError(
                f'{model_method_schema.method_name} is not a method in {self}.'
            )

        self.model_method_schema_dict[model_method_schema.method_name] = model_method_schema

    def set_model_method_input_schema(self, method_name: str, data_frame_schema: DataFrameSchema):
        if not hasattr(self, 'model_method_schema_dict'):
            self.model_method_schema_dict = {}

        if method_name not in self.model_method_schema_dict:
            model_method_schema = ModelMethodSchema(method_name, input_schema=data_frame_schema)
            self.model_method_schema_dict[method_name] = model_method_schema
        else:
            self.model_method_schema_dict[method_name].set_input_schema(data_frame_schema)

    def set_model_method_output_schema(self, method_name: str, data_frame_schema: DataFrameSchema):
        if not hasattr(self, 'model_method_schema_dict'):
            self.model_method_schema_dict = {}

        if method_name not in self.model_method_schema_dict:
            model_method_schema = ModelMethodSchema(method_name, output_schema=data_frame_schema)
        else:
            self.model_method_schema_dict[method_name].set_output_schema(data_frame_schema)


    # def get_open_api_yaml(self):
    #     """Get the open API spec for the the model predictions in a YAML representation.
    #
    #     Returns
    #     -------
    #     str
    #        YAML representation of the open API spec for the the model predictions
    #     """
    #     return open_api_yaml_specification(
    #         feature_df_schema=self.feature_df_schema,
    #         target_df_schema=self.target_df_schema
    #     )
    #
    # def get_open_api_dict(self):
    #     """Get the open API spec for the the model predictions in a dictionary representation.
    #
    #     Returns
    #     -------
    #     dict
    #        Dictionary representation of the open API spec for the the model predictions
    #     """
    #     return open_api_dict_specification(
    #         feature_df_schema=self.feature_df_schema,
    #         target_df_schema=self.target_df_schema
    #     )


def _validate_model_method_column_info(model_method_column_info):
    if isinstance(model_method_column_info, ModelMethodColumnInfo) is False:
        raise TypeError(f'model_method_column_info must be an instance of ModelMethodColumnInfo')


def _validate_model_method_schema(model_method_schema):
    if isinstance(model_method_schema, ModelMethodSchema) is False:
        raise TypeError(f'model_method_schema must be an instance of ModelMethodSchema')


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def infer_feature_df_schema_from_fit(method_list: Union[None, List[str]] = None):
    if method_list is None:
        method_list = ['predict']

    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            self_var = args[0]
            X = args[1]
            y = args[2]

            if isinstance(X, pd.DataFrame) is False:
                raise ValueError(
                    "X must be a pandas DataFrame."
                )

            if isinstance(self_var, DataFrameModel) is False:
                raise ValueError(
                    "Class must inherit from DataFrameModelMixin."
                )

            for method_name in method_list:
                if method_name not in self_var.model_method_column_info_dict:
                    input_df_schema = get_data_frame_schema_from_df(X)
                else:
                    model_method_column_info = self_var.model_method_column_info_dict[method_name]
                    input_df_schema = get_data_frame_schema_from_df(
                        X,
                        enum_columns=model_method_column_info.input_enum_columns,
                        interval_columns=model_method_column_info.input_interval_columns,
                        interval_buffer_percent=model_method_column_info.input_interval_percent_buffer,
                    )
                self_var.set_model_method_input_schema(method_name, input_df_schema)

            return func(*args)
        return wrapper
    return decorator


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

        if isinstance(self_var, DataFrameModel) is False:
            raise ValueError(
                "Class must inherit from DataFrameModelMixin."
            )

        if 'predict' not in self_var.model_method_column_info_dict:
            output_df_schema = get_data_frame_schema_from_df(y)
        else:
            model_method_column_info = self_var.model_method_column_info_dict['predict']
            output_df_schema = get_data_frame_schema_from_df(
                y,
                enum_columns=model_method_column_info.output_enum_columns,
            )
        self_var.set_model_method_output_schema('predict', output_df_schema)

        return func(*args)

    return wrapper


def validate_prediction_input_and_output(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        df = args[1]

        if isinstance(df, pd.DataFrame) is False:
            raise ValueError(
                "First argument of function must be a pandas DataFrame."
            )

        df = self_var.model_method_schema_dict[func.__name__].input_schema.validate_df(df)

        return_values = func(*args)

        return_values = self_var.model_method_schema_dict[func.__name__].output_schema.validate_df(return_values)

        return return_values

    return wrapper


########################################################################################################
# Data frame model that uses separate model for each category of a feature.
########################################################################################################
# class FeatureSplitModel(BaseModel, DataFrameModelMixin):
#     MODEL_NAME = 'Feature split meta model'
#
#     def __init__(
#             self,
#             features,
#             group_column,
#             group_model_dict,
#     ):
#         super().__init__()
#         self.features = features
#         self.group_column = group_column
#         self.group_model_dict = group_model_dict
#         self.target_columns = None
#
#     @infer_target_df_schema_from_fit(infer_enums=False)
#     @infer_feature_df_schema_from_fit(infer_enums=False, infer_intervals=False)
#     def fit(self, X, y):
#         assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
#         assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
#
#         for group in X[self.group_column].unique():
#             mask = (X[self.group_column] == group)
#             self.group_model_dict[group].fit(X[mask], y[mask])
#
#         self.target_columns = y.columns
#
#     # @validate_prediction_input_and_output
#     def predict(self, X):
#         assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
#         assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"
#         X = X.copy()
#
#         X.append(pd.Series(name='prediction'))
#         for group in X[self.group_column].unique():
#             mask = (X[self.group_column] == group)
#             X.loc[mask, 'prediction'] = self.group_model_dict[group].predict(X[mask])
#             prediction_df = pd.DataFrame(data=X['prediction'], columns=self.target_columns)
#         return prediction_df


# ########################################################################################################
# # Wrapper for mlflow
# ########################################################################################################
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)
