from pathlib import Path
import mlflow.pyfunc


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
    def predict_from_dict(cls, data_dict):
        prediction = cls.model.python_model.model.predict(cls.model.python_model.model.model_input_from_dict(data_dict))

        # Return prediction json response
        response = cls.model.python_model.model.model_output_to_json(prediction)
        return response
