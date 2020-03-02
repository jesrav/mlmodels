# WIP

## Installation
```bash
pip install git+https://github.com/jesrav/mlmodels#egg=mlmodels
```
To install dependencies for examles
```bash
pip install -r examples/requirements.txt
```
## Base class for ML models
The BaseModel class is an abstract class that enforces child classes to implement
- A MODEL_NAME attribute
- A fit method
- A predict method

It gives you the time the model was fitted plus serialization and deserialization out of the box.

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

# When was the model initialized?
dummy_model.model_initiated_dt
# Returns:  datetime.datetime(2020, 2, 12, 9, 46, 19, 81250)

# Predict
loaded_model.predict([[1, 1], [2, 2]])
```
## Data frame model mixin class
If you create a model that takes a data frame and outputs a data frame when predicting, you can use the DataFrameModelMixin class to add some functionality.
It adds the following methods: 
- set_feature_df_schema: setting the schema of the schema of the model input 
- set_target_df_schema: setting the schema of the prediction data frame.
- get_open_api_yaml/get_open_api_dict: Generating an open api specification.

The DataFrameModelMixin class can be used in combination with the accompanying decorators to infer the feature and target schema on fit and validate new data on predict.

### Example use
```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlmodels import (
    BaseModel,
    DataFrameModelMixin,
    infer_target_df_schema_from_fit,
    infer_feature_df_schema_from_fit,
    validate_prediction_input_and_output
)

class RandomForestClassifierModel(BaseModel, DataFrameModelMixin):
    MODEL_NAME = 'Random forest classifier model'

    def __init__(
            self,
            features,
            feature_enum_columns=None,
            target_enum_columns=None,
            feature_interval_columns=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30},
    ):
        super().__init__()
        self.features = features
        self.target_columns = None,
        self.feature_enum_columns = feature_enum_columns
        self.target_enum_columns = target_enum_columns
        self.feature_interval_columns = feature_interval_columns
        self.random_forest_params = random_forest_params
        self.model = RandomForestClassifier(**random_forest_params)

    @infer_feature_df_schema_from_fit(infer_enums=True, infer_intervals=True, interval_buffer_percent=15)
    @infer_target_df_schema_from_fit(infer_enums=True)
    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        self.target_columns = y.columns
        return self

    @validate_prediction_input_and_output
    def predict(self, X):
        predictions_array = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(data=predictions_array, columns=self.target_columns)
        return predictions_df

# Read data
csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(csv_url, sep=';')

# Create 3 randomly assigned groups
data['group1'] = np.random.choice(3, len(data))
data['group2'] = np.random.choice([3, 7], len(data))
data['group1'] = data['group1'].astype('int64')
data['group2'] = data['group2'].astype('int64')

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# Fit model, make predictions and evaluate
model = RandomForestClassifierModel(
    features=train_x.columns,
    feature_enum_columns=['group1', 'group2'],
    random_forest_params={'n_estimators': 100, 'max_depth': 15},
)
model.fit(train_x, train_y)

predicted_qualities = model.predict(test_x)
```
### Model input schema validation
If the input dataframe does not match the data frame schema you will get an error.
```python
# Example of missing features
model.predict(test_x[["density", "chlorides", "alcohol"]])
# returns: pandera.errors.SchemaError: column 'fixed acidity' not in dataframe

# Example of wrong dtype
test_x_copy = test_x.copy()
test_x_copy.density = test_x_copy.density.astype('int64')
model.predict(test_x_copy)
# returns: pandera.errors.SchemaError: expected series 'density' to have type float64, got int64

# Example of wrong categorical value/enum.
test_x_copy = test_x.copy()
test_x_copy.group1 = 100
model.predict(test_x_copy)
# returns: pandera.errors.SchemaError: <Schema Column: 'group1' type=int64> failed element-wise validator 0:
# <Check _isin: isin(frozenset({0, 1, 2}))>
```

## Creating MLFLOW pyfunc model
You can wrap your model with the MLFlowWrapper class, to make your model comply with the mlflow model format.
```python
from mlmodels import MLFlowWrapper
mlflow_model = MLFlowWrapper(model)
```

### Building a docker image with a model service
The model must wrapped as an mlflow.pyfunc model and must inherit from the BaseModel and DataFrameModelMixin

First we train and save a model locally. You need to use python 3.6.7 or update the python version in examples/random_forest_model_example/conda.yaml.
```console
python examples/random_forest_model_example/wine_example.py
```
Then we build a docker image for serving the model as a web API. 
```console
mlmodels dockerize examples/random_forest_model_example/model_output/wine_model 1 model-service:latest
```
To run the model service locally
```console
docker run -p 5000:5000 model-service:latest
```
The swagger specification can be found at http://localhost:5000/apidocs/
![](docs/swagger_screenshot.jpg)