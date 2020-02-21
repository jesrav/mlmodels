import pickle as pickle
from datetime import datetime
from abc import ABCMeta, abstractmethod
from functools import wraps
import numpy as np
import pandas as pd
import mlflow.pyfunc
import json

from mlmodels.openapi_yaml_template import open_api_yaml_specification, open_api_dict_specification


########################################################################################################
# Base model classes
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


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def data_frame_model(infer_feature_dtypes=True, infer_target_dtypes=True):
    """

    Parameters
    ----------
    cls
        Class that is modified
    infer_feature_dtypes

    infer_target_dtypes

    Returns
    -------

    """
    def outer_func(cls):
        @wraps(cls)
        def wrapper(*args, **kws):

            if not (
                {'features', 'categorical_columns'}
                .issubset(set(cls.__init__.__code__.co_varnames))
            ):
                raise ValueError("features' and 'categorical_columns' must be initialized in model class")


            cls.ACCEPTED_DTYPES = (
                np.dtype('int64'),
                np.dtype('int32'),
                np.dtype('float64'),
                np.dtype('float32'),
                np.dtype('O'),
            )

            cls.DTYPE_TO_JSON_TYPE_MAP = {
                np.dtype('int64'): {'type': 'number', 'format': 'integer'},
                np.dtype('int32'): {'type': 'number', 'format': 'integer'},
                np.dtype('float64'): {'type': 'number', 'format': 'float'},
                np.dtype('float32'): {'type': 'number', 'format': 'float'},
                np.dtype('O'): {'type': 'string'},
            }

            def set_feature_dtypes(self, feature_dtype_dict):
                """Set the dtypes of the features the model takes as input

                Parameters
                ----------
                feature_dtype_dict
                    Dictionary that maps the feature names to their dtypes.
                    Accepted dtypes can be found in the ACCEPTED_DTYPES attribute.

                Examples
                --------
                dtype_dict = {
                            'hej': np.dtype('int64'),
                            'nej': np.dtype('int32'),
                }
                model.set_feature_dtypes(dtype_dict)
                """

                if not isinstance(feature_dtype_dict, dict):
                    raise TypeError('feature_dtypes must be a dictionary')
                if not set(feature_dtype_dict.keys()) == set(self.features):
                    raise TypeError('feature_dtypes must have a key for all the features')
                if not all([isinstance(feature_dtype_dict[k], np.dtype) for k in feature_dtype_dict]):
                    raise TypeError(
                        'All the values in the dictionary feature_dtypes must be of type np.dtype'
                    )
                if not all([(feature_dtype_dict[k] in self.ACCEPTED_DTYPES) for k in feature_dtype_dict]):
                    raise TypeError(
                        f'The values in the dictionary feature_dtypes can only be the one \
                         of the accepted dtypes: {self.ACCEPTED_DTYPES}'
                    )

                self.feature_dtypes = pd.Series(feature_dtype_dict)

            def set_target_dtypes(self, target_dtype_dict):
                """Set the dtypes of the output of the model

                Parameters
                ----------
                target_dtype_dict
                    Dictionary that maps the target names to their dtypes.
                    Accepted dtypes can be found in the ACCEPTED_DTYPES attribute.

                Examples
                --------
                dtype_dict = {
                            'hej': np.dtype('int64'),
                            'nej': np.dtype('int32'),
                }
                model.set_target_dtypes(dtype_dict)
                """

                if not isinstance(target_dtype_dict, dict):
                    raise TypeError('feature_dtypes must be a dictionary')
                if not all([isinstance(target_dtype_dict[k], np.dtype) for k in target_dtype_dict]):
                    raise TypeError(
                        'All the values in the dictionary feature_dtypes must be of type np.dtype'
                    )
                if not all([(target_dtype_dict[k] in self.ACCEPTED_DTYPES) for k in target_dtype_dict]):
                    raise TypeError(
                        f'The values in the dictionary feature_dtypes can only be the one \
                         of the accepted dtypes: {self.ACCEPTED_DTYPES}'
                    )

                self.target_dtypes = pd.Series(target_dtype_dict)

            def get_model_record_field_schemas(self):
                out_dict = {
                    'features': self.feature_dtypes.apply(lambda x: self.DTYPE_TO_JSON_TYPE_MAP[x]).to_dict(),
                    'targets': self.target_dtypes.apply(lambda x: self.DTYPE_TO_JSON_TYPE_MAP[x]).to_dict()
                }
                return out_dict

            def get_open_api_yaml(self):
                return open_api_yaml_specification(
                    model_input_record_field_schema_dict=self.get_model_input_record_field_schema()['features'],
                    possible_categorical_column_values=(self.possible_categorical_column_values or {}),
                    model_target_field_schema_dict=self.get_model_input_record_field_schema()['targets']
                )

            def get_open_api_dict(self):
                if not hasattr(self, 'possible_categorical_column_values'):
                    self.possible_categorical_column_values = None
                return open_api_dict_specification(
                    model_input_record_field_schema_dict=self.get_model_input_record_field_schema()['features'],
                    possible_categorical_column_values=(self.possible_categorical_column_values or {}),
                    model_target_field_schema_dict=self.get_model_input_record_field_schema()['targets']
                )

            def convert_model_input_dtypes(self, model_input):
                """If types inferred py pandas to not match the required dtypes,
                we try to convert them."""
                dtype_dict = self.feature_dtypes.astype(str).to_dict()
                return model_input.astype(dtype_dict)

            def model_input_from_dict(self, dict_data):
                """Read data from record type dictionary representation."""

                model_input = pd.DataFrame.from_records(dict_data['data'])
                return self.convert_model_input_dtypes(model_input)

            def model_output_to_json(self, model_predictions):
                """Transform model predictions to record type json representation."""
                if isinstance(model_predictions, pd.Series):
                    return model_predictions.to_frame().to_json(orient='records')
                elif isinstance(model_predictions, pd.DataFrame):
                    return model_predictions.to_dict(orient='records')
                elif isinstance(model_predictions, np.ndarray):
                    return json.dumps({'predictions': model_predictions.tolist()})
                else:
                    return json.dumps({'predictions': model_predictions})

            # Set new class methods
            cls.get_model_input_record_field_schema = get_model_record_field_schemas
            cls.get_open_api_yaml = get_open_api_yaml
            cls.get_open_api_dict = get_open_api_dict
            cls.convert_model_input_dtypes = convert_model_input_dtypes
            cls.model_input_from_dict = model_input_from_dict
            cls.model_output_to_json = model_output_to_json
            cls.set_feature_dtypes = set_feature_dtypes
            cls.set_target_dtypes = set_target_dtypes

            # Modify class methods
            if infer_feature_dtypes and not infer_target_dtypes:
                cls.fit = infer_dtypes_from_fit(infer_target_dtypes=False)(cls.fit)
            if infer_feature_dtypes and infer_target_dtypes:
                cls.fit = infer_dtypes_from_fit(infer_target_dtypes=True)(cls.fit)
            cls.predict = validate_prediction_input_schema(cls.predict)
            cls.fit = infer_category_feature_values_from_fit(cls.fit)

            return cls(*args, **kws)

        return wrapper

    return outer_func


def infer_dtypes_from_fit(infer_target_dtypes=True):

    def outer_func(func):
        @wraps(func)
        def wrapper(*args):
            self_var = args[0]
            X = args[1]
            y = args[2]

            if isinstance(X, pd.DataFrame) is False:
                raise ValueError(
                    "X must be a pandas DataFrame."
                )

            if (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)) is False:
                raise ValueError(
                    "y must be a pandas Series or DataFrame."
                )

            if self_var.features is None:
                raise ValueError("features attribute must be set. It should be a list of features")

            if all(X[self_var.features].dtypes.isin(self_var.ACCEPTED_DTYPES)):
                self_var.feature_dtypes = X[self_var.features].dtypes
            else:
                raise ValueError(f"Dtypes of columns of X must be in {self_var.ACCEPTED_DTYPES}]")

            if infer_target_dtypes:
                if isinstance(y, pd.Series):
                    y_dtypes = y.to_frame().dtypes
                else:
                    y_dtypes = y.dtypes

                if all(y_dtypes.isin(self_var.ACCEPTED_DTYPES)):
                    self_var.target_dtypes = y_dtypes
                else:
                    raise ValueError(f"Dtypes of y must be in {self_var.ACCEPTED_DTYPES}]")

            return func(*args)

        return wrapper
    return outer_func


def infer_category_feature_values_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if self_var.categorical_columns is None:
            return func(*args)
        else:

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

            if not set(self_var.categorical_columns).issubset(set(self_var.features).union({y.name})):
                raise ValueError("categorical_features must be a subset of the union of features and target name")

            categorical_features = [
                cat_feat for cat_feat in self_var.categorical_columns
                if cat_feat in self_var.features
            ]
            categorical_target = [
                cat_feat for cat_feat in self_var.categorical_columns
                if cat_feat == y.name
            ][0]

            self_var.possible_categorical_column_values = {
                categorical_feature: list(X[categorical_feature].unique())
                for categorical_feature in categorical_features
            }

            if categorical_target:
                self_var.possible_categorical_column_values[categorical_target] = list(y.unique())

            return func(*args)

    return wrapper


def infer_dataframe_features_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        self_var.features = list(X.columns)

        return func(*args)

    return wrapper


def validate_prediction_input_schema(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]

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


# def categorical_feature_valid(series, options):
#     return all(series.isin(options))
#
#
# def get_categorical_features_validity_dict(df, categorical_features, categorical_feature_options):
#     categorical_feature_valid_dict = {
#         categorical_feature: categorical_feature_valid(df[categorical_feature], categorical_feature_options)
#         for categorical_feature in categorical_features
#     }
#     return categorical_feature_valid_dict
#
#
# def categorical_features_valid(df, categorical_features, categorical_feature_options):
#     categorical_feature_valid_dict = get_categorical_features_validity_dict(df, categorical_features, categorical_feature_options)
#     return all(categorical_feature_valid_dict[k] for k in categorical_feature_valid_dict)
#
# def validate_prediction_input_category_values(func):
#
#     @wraps(func)
#     def wrapper(*args):
#         self_var = args[0]
#         X = args[1]
#
#         if isinstance(X, pd.DataFrame) is False:
#             raise ValueError(
#                 "X must be a pandas DataFrame."
#             )
#
#         if not categorical_features_valid(X, self_var.categorical_columns, self_var.possible_categorical_column_values):
#             categorical_feature_valid_dict = get_categorical_features_valid_dict(
#                 X,
#                 self_var.categorical_columns,
#                 self_var.possible_categorical_column_values
#             )
#             categorical_features_not_valid_list =
#
#             raise ValueError("features attribute must be set. It should be a list of features")
#
#         return_values = func(*args)
#         return return_values
#
#     return wrapper

########################################################################################################
# Data frame model that uses seperate model for ecah category of a feature.
########################################################################################################
@data_frame_model
class FeatureSplitModel(BaseModel):
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

    def fit(self, X, y):
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        assert self.group_column in X.columns, f"{self.group_column} must be a columns in X"

        for group in X[self.group_column].unique():
            mask = (X[self.group_column] == group)
            self.group_model_dict[group].fit(X[mask], y[mask])

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