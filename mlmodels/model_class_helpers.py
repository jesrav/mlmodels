import pandas as pd
from mlmodels import DataFrameModel, infer_dataframe_dtypes_from_fit, infer_dataframe_features_from_fit

class FeatureSplitModel(DataFrameModel):
    MODEL_NAME = 'Feature split meta model'

    def __init__(self, group_column=None, group_model_dict=None):
        super().__init__()
        self.group_model_dict = group_model_dict
        self.group_column = group_column

    @infer_dataframe_features_from_fit
    @infer_dataframe_dtypes_from_fit
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