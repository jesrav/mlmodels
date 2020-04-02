from functools import wraps
from typing import Union, List

import mlflow.pyfunc
import pandas as pd

from mlmodels.data_frame_schema import (
    DataFrameSchema,
    get_data_frame_schema_from_df)
from mlmodels.openapi_spec import (
    open_api_yaml_specification_from_df_method,
    open_api_dict_specification_from_df_method
)
from mlmodels import BaseModel

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
class DataFrameModelMixin:
    """Data frame model class

    The data frame model mixin class can be used to add functionality to a model class that has methods that
    take a Pandas DataFrame as input and produces output in the form of a Pandas DataFrame.
    """

    def set_model_method_column_info(
            self,
            model_method_column_info: ModelMethodColumnInfo
    ):

        # _validate_model_method_column_info(model_method_column_info)

        if not hasattr(self, 'model_method_column_info_dict'):
            self.model_method_column_info_dict = {}

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

        if not hasattr(self, 'model_method_schema_dict'):
            self.model_method_schema_dict = {}

        if model_method_schema.method_name in self.model_method_schema_dict:
            raise ValueError(f'{model_method_schema.method_name} has already been set.')

        self_methods = [
            method_name for method_name in dir(self)
            if callable(getattr(self, method_name))
        ]

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
            self.model_method_schema_dict[method_name] = model_method_schema
        else:
            self.model_method_schema_dict[method_name].set_output_schema(data_frame_schema)

    def get_method_open_api_yaml(self, method_name):
        """Get the open API spec for the the model predictions in a YAML representation.

        Returns
        -------
        str
           YAML representation of the open API spec for the the model predictions
        """
        return open_api_yaml_specification_from_df_method(
            method_name,
            self.model_method_schema_dict[method_name].input_schema,
            self.model_method_schema_dict[method_name].output_schema,
        )

    def get_method_open_api_dict(self, method_name):
        """Get the open API spec for the the model predictions in a dictionary representation.

        Returns
        -------
        dict
           Dictionary representation of the open API spec for the the model predictions
        """
        return open_api_dict_specification_from_df_method(
            method_name,
            self.model_method_schema_dict[method_name].input_schema,
            self.model_method_schema_dict[method_name].output_schema,
        )


def _validate_model_method_column_info(model_method_column_info):
    if isinstance(model_method_column_info, ModelMethodColumnInfo) is False:
        raise TypeError(f'model_method_column_info must be an instance of ModelMethodColumnInfo')


def _validate_model_method_schema(model_method_schema):
    if isinstance(model_method_schema, ModelMethodSchema) is False:
        raise TypeError(f'model_method_schema must be an instance of ModelMethodSchema')


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def infer_feature_df_schema_from_fit(method_list: Union[None, List[str]]):

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

            if isinstance(self_var, DataFrameModelMixin) is False:
                raise ValueError(
                    "Class must inherit from DataFrameModelMixin."
                )

            for method_name in method_list:
                if not hasattr(self_var, 'model_method_column_info_dict'):
                    input_df_schema = get_data_frame_schema_from_df(X)
                elif method_name not in self_var.model_method_column_info_dict:
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

        if isinstance(self_var, DataFrameModelMixin) is False:
            raise ValueError(
                "Class must inherit from DataFrameModelMixin."
            )

        if not hasattr(self_var, 'model_method_column_info_dict'):
            output_df_schema = get_data_frame_schema_from_df(y)
        elif 'predict' not in self_var.model_method_column_info_dict:
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


def validate_method_input_and_output(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        df = args[1]

        if isinstance(df, pd.DataFrame) is False:
            raise ValueError(
                "First argument of function must be a pandas DataFrame."
            )

        _ = self_var.model_method_schema_dict[func.__name__].input_schema.validate_df(df)

        return_values = func(*args)

        return_values = self_var.model_method_schema_dict[func.__name__].output_schema.validate_df(return_values)

        return return_values

    return wrapper


def infer_from_fit(
    feature_df_schema: bool,
    target_df_schema: bool,
    methods_with_features_as_input: Union[None, List[str]] = None,
    validate_input_output_method_list: Union[None, List[str]] = None,
):
    if feature_df_schema and methods_with_features_as_input is None:
        raise ValueError('If feature_df_schema is True then a list of methods must be passed.')

    def decorator(cls):
        @wraps(cls)
        def wrapper(*args, **kws):

            # Modify fit method.
            cls.fit = infer_feature_df_schema_from_fit(methods_with_features_as_input)(cls.fit)
            cls.fit = infer_target_df_schema_from_fit(cls.fit)

            # Modify methods where input and output should be validated.
            if validate_input_output_method_list is not None:
                for method_name in validate_input_output_method_list:
                    method = getattr(cls, method_name)
                    method = validate_method_input_and_output(method)
                    setattr(cls, method_name, method)

            return cls(*args, **kws)

        return wrapper
    return decorator

# ########################################################################################################
# # Wrapper for mlflow
# ########################################################################################################
class SKLearnWrapper(BaseModel, DataFrameModelMixin):

    def __init__(
            self,
            features,
            sklearn_model_class,
            sklearn_model_params,
    ):
        super().__init__()
        self.features = features
        self.target_columns = None
        self.model = sklearn_model_class(**sklearn_model_params)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df

# ########################################################################################################
# # Wrapper for mlflow
# ########################################################################################################
class MLFlowWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

