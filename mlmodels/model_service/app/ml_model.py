from typing import Dict
from pathlib import Path
import pandas as pd
import mlflow.pyfunc
import json


class MLModel:
    """Singleton class to hold the ml model"""
    model = None
    model_version = None
    open_api_dict = None

    @classmethod
    def load_model(cls):
        model_path = Path('app/model')
        cls.model = mlflow.pyfunc.load_model(str(model_path))
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
    def model_input_from_dict(cls, model_input_dict: Dict) -> pd.DataFrame:
        """Read data from record type dictionary representation.

        The data frame dtypes are changed to the model schema. The reason this is not done is that
        the decoding of json changes a float like 1.0 to an integer.


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
        model_input_modified = model_input.astype(
            cls.model.python_model.model.feature_df_schema.get_dtypes()
        )
        return cls.model.python_model.model.feature_df_schema.validate_df(model_input_modified)

    @staticmethod
    def model_output_to_json(model_predictions: pd.DataFrame) -> str:
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

        if not isinstance(model_predictions, pd.DataFrame):
            raise TypeError('model_predictions must be a pandas DataFrame')

        return str(model_predictions.to_dict(orient='records'))

    @classmethod
    def predict_from_dict(cls, data_dict):
        prediction = cls.model.python_model.model.predict(cls.model_input_from_dict(data_dict))

        # Return prediction json response
        response = cls.model_output_to_json(prediction)
        return response


