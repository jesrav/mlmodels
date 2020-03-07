from typing import Dict
from pathlib import Path
import pandas as pd
import mlflow.pyfunc

MODEL_PATH = Path('app/model')

class MLModel:
    """Singleton class to hold the ml model"""
    model = None
    model_initiated_dt = None
    model_version = None
    open_api_dict = None

    @classmethod
    def load_model(cls):
        cls.model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        cls.model_initiated_dt = cls.model.python_model.model.model_initiated_dt

    @classmethod
    def set_model_version(cls, model_version):
        cls.model_version = model_version

    @classmethod
    def get_model_version(cls):
        return cls.model_version

    @classmethod
    def get_model_train_datetime(cls):
        return cls.model_initiated_dt

    @classmethod
    def model_method_input_from_dict(cls, method: str, input_dict: Dict) -> pd.DataFrame:
        """Read data from record type dictionary representation.

        The data frame dtypes are changed to the model schema. The reason this is not done is that
        the decoding of json changes a float like 1.0 to an integer.


        Parameters
        ----------
        method: str
            Name of model method.
        input_dict: dict
            dictionary with the data frame data in a record type representation.

        Returns
        -------
        pandas DataFrame
            Data frame with model input.
        """

        input = pd.DataFrame.from_records(input_dict['data'])
        input_df_schema = cls.model.python_model.model.model_method_schema_dict[method].input_schema
        model_input_modified = input.astype(
            input_df_schema.get_dtypes()
        )
        return input_df_schema.validate_df(model_input_modified)

    @staticmethod
    def df_to_json(df: pd.DataFrame) -> str:
        """Transform model predictions to record type json representation.

        Parameters
        ----------
        df: Pandas DataFrame.

        Raises
        ------
        TypeError
            If model_predictions is not a pandas Series or DataFrame

        Returns
        -------
        str
            String with json representation of model predictions
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError('model_predictions must be a pandas DataFrame')

        return str(df.to_dict(orient='records'))

    @classmethod
    def method_call_from_dict_input(cls, method: str, data_dict: dict) -> str:
        model_method = cls.model.python_model.model.__getattribute__(method)
        method_output = model_method(
            cls.model_method_input_from_dict(method, data_dict)
        )

        # Return prediction json response
        response = cls.df_to_json(method_output)
        return response


