# WIP

## Installation
To install as package
```bash
pip install git+https://github.com/jesrav/mlmodels#egg=mlmodels
```
To install dependencies for examles
```bash
pip install -r requirements.txt
```
## Base classes for ML models
The BaseModel class is an abstract class that enforces child classes to implement
- A MODEL_NAME attribute
- A fit method
- A predict method
It gives you serialization and deserialization out of the box.

```python
from mlmodels import BaseModel

class DummyModel(BaseModel):
    MODEL_NAME = 'Dummy model'
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        return len(X)*[1]

dummy_model = DummyModel()

# Save model
dummy_model.save(fname='model.pickle')

# Load model
loaded_model = DummyModel().load('model.pickle')

# Predict
loaded_model.predict([[1, 1], [2, 2]])
```
## DataFrameModel class
The DataFrameModel class inherits from BaseModel and is meant to be used with structured data in pandas dataframes.
It has some methods for using the features and dtypes of the input dataframe to generate an open api specification.
It also has some helper decorators for inferring features and dtypes on fit and validating input on predict.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mlmodels import DataFrameModel, infer_dataframe_dtypes_from_fit, validate_prediction_input

# Create model class
class RandomForestRegressorModel(DataFrameModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30}
    ):
        super().__init__()
        self.features = features
        self.random_forest_params = random_forest_params
        self.model = RandomForestRegressor(**random_forest_params)

    @infer_dataframe_dtypes_from_fit
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self.model

    @validate_prediction_input
    def predict(self, X):
        predictions = self.model.predict(X[self.features])
        return predictions

# Read data
csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(csv_url, sep=';')

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train["quality"]
test_y = test["quality"]

# Fit model make predictions and evaluate
features = ["pH", "density", "chlorides", "alcohol"]
model = RandomForestRegressorModel(
    features=features,
    random_forest_params={'n_estimators': 100, 'max_depth': 15},
)
model.fit(train_x, train_y)

predicted_qualities = model.predict(test_x)
```
### Model input schema
If the input dataframe does not have the right features or the columns do not have the right dtypes,
you will get an error.
```python
# Example of missing feature
model.predict(test_x[["density", "chlorides", "alcohol"]])
# returns: ValueError: The following features must be in X: ['pH', 'density', 'chlorides', 'alcohol']

# Example of wrong dtype
test_x.density = test_x.density.astype('int64')
model.predict(test_x)
# returns: ValueError: Dtypes must be: {'pH': dtype('float64'), 'density': dtype('float64'), 'chlorides': dtype('float64'), 'alcohol': dtype('float64')}
```
You can get the open api spec for the model in either yaml or as a dictionary, using the record-orientation of pandas.
```python
from pprint import pprint
pprint(model.get_open_api_dict())
# Returns:
# {'definitions': {'prediction_input': {'properties': {'data': {'items': {'properties': {'alcohol': {'format': 'float',
#                                                                                                    'nullable': False,
#                                                                                                    'type': 'number'},
#                                                                                        'chlorides': {'format': 'float',
#                                                                                                      'nullable': False,
#                                                                                                      'type': 'number'},
#                                                                                        'density': {'format': 'float',
#                                                                                                    'nullable': False,
#                                                                                                    'type': 'number'},
#                                                                                        'pH': {'format': 'float',
#                                                                                               'nullable': False,
#                                                                                               'type': 'number'}}},
#                                                               'type': 'array'}},
#                                       'required': ['data'],
#                                       'type': 'object'},
#                  'predictions': {'properties': {'predictions': {'items': {'format': 'integer',
#                                                                           'nullable': False,
#                                                                           'type': 'number'},
#                                                                 'type': 'array'},
#                                                 'type': 'object'}}},
#  'info': {'title': 'Prediction open api spec', 'version': '1.0.0'},
#  'openapi': '3.0.2',
#  'parameters': [{'description': 'List of feature records.',
#                  'in': 'body',
#                  'name': 'data',
#                  'required': True,
#                  'schema': {'$ref': '#/definitions/prediction_input'}}],
#  'responses': {200: {'description': 'List of predictions',
#                      'name': 'predictions',
#                      'schema': {'$ref': '#/definitions/predictions'}}},
#  'tags': ['ml_endpoint']}
```

## Creating MLFLOW pyfunc model
You can wrap your model with the MLFlowWrapper class, to make your model comply with the mlflow model format.
```python
from mlmodels import MLFlowWrapper
mlflow_model = MLFlowWrapper(model)
```

