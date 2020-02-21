from functools import wraps
import numpy as np
import pandas as pd
import mlflow.pyfunc

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
        self.features=None
        self.feature_dtypes=None
        self.target_dtypes=None
        self.possible_categorical_column_values=None
        self.categorical_columns = None

    ACCEPTED_DTYPES = (
        np.dtype('int64'),
        np.dtype('int32'),
        np.dtype('float64'),
        np.dtype('float32'),
        np.dtype('O'),
    )

    _DTYPE_TO_JSON_TYPE_MAP = {
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

    def _get_model_record_field_schemas(self):
        out_dict = {
            'features': self.feature_dtypes.apply(lambda x: self._DTYPE_TO_JSON_TYPE_MAP[x]).to_dict(),
            'targets': self.target_dtypes.apply(lambda x: self._DTYPE_TO_JSON_TYPE_MAP[x]).to_dict()
        }
        return out_dict

    def get_open_api_yaml(self):
        """Get the open API spec for the the model predictions in a YAML representation.

        Returns
        -------
        str
           YAML representation of the open API spec for the the model predictions
        """
        return open_api_yaml_specification(
            model_input_record_field_schema_dict=self._get_model_record_field_schemas()['features'],
            possible_categorical_column_values=(self.possible_categorical_column_values or {}),
            model_target_field_schema_dict=self._get_model_record_field_schemas()['targets']
        )

    def get_open_api_dict(self):
        """Get the open API spec for the the model predictions in a dictionary representation.

        Returns
        -------
        dict
           Dictionary representation of the open API spec for the the model predictions
        """
        if not hasattr(self, 'possible_categorical_column_values'):
            self.possible_categorical_column_values = None
        return open_api_dict_specification(
            model_input_record_field_schema_dict=self._get_model_record_field_schemas()['features'],
            possible_categorical_column_values=(self.possible_categorical_column_values or {}),
            model_target_field_schema_dict=self._get_model_record_field_schemas()['targets']
        )

    def _convert_model_input_dtypes(self, model_input):
        """If types inferred py pandas to not match the required dtypes,
        we try to convert them."""
        dtype_dict = self.feature_dtypes.astype(str).to_dict()
        return model_input.astype(dtype_dict)

    def model_input_from_dict(self, model_input_dict):
        """Read data from record type dictionary representation.

        Parameters
        ----------
        model_input_dict: dict
            dictionary with the data frame data in a record type representation.

        Returns
        -------
        pandas DataFrame
            Data frame with model input.
        """

        model_input = pd.DataFrame.from_records(model_input_dict['data'])
        return self._convert_model_input_dtypes(model_input)

    @staticmethod
    def model_output_to_json(model_predictions):
        """Transform model predictions to record type json representation.

        Parameters
        ----------
        model_predictions

        Raises
        ------
        TypeError
            If model_predictions is not a pandas Series or DataFrame

        Returns
        -------
        str
            String with json representation of model predictions
        """
        if isinstance(model_predictions, pd.Series):
            return model_predictions.to_frame().to_json(orient='records')
        elif isinstance(model_predictions, pd.DataFrame):
            return model_predictions.to_dict(orient='records')
        else:
            raise TypeError('model_predictions must be a pandas Series or DataFrame')


########################################################################################################
# Decorators to help create models from DataFrameModel class.
########################################################################################################
def infer_feature_dtypes_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if isinstance(X, pd.DataFrame) is False:
            raise ValueError(
                "X must be a pandas DataFrame."
            )

        if self_var.features is None:
            raise ValueError("features attribute must be set. It should be a list of features")

        if all(X[self_var.features].dtypes.isin(self_var.ACCEPTED_DTYPES)):
            self_var.feature_dtypes = X[self_var.features].dtypes
        else:
            raise ValueError(f"Dtypes of columns of X must be in {self_var.ACCEPTED_DTYPES}]")

        return func(*args)

    return wrapper


def infer_target_dtypes_from_fit(func):

    @wraps(func)
    def wrapper(*args):
        self_var = args[0]
        X = args[1]
        y = args[2]

        if (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)) is False:
            raise ValueError(
                "y must be a pandas Series or DataFrame."
            )

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


def infer_category_values_from_fit(func):

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

            if (isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)) is False:
                raise ValueError(
                    "y must be a pandas Series or DataFrame."
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

    @validate_prediction_input_schema
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
