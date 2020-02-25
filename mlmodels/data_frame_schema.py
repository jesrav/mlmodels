from typing import List, Dict
from collections import namedtuple
import pandas as pd
import pandera as pa

_ACCEPTED_DTYPES = (
    'int64',
    'float64',
    'object',
    'string',
)

_dtype_to_pandera_map = {
    'int64': pa.Int,
    'float64': pa.Float,
    'object': pa.String,
    'string': pa.String,
}


def _validate_name(name):
    if not isinstance(name, str):
        raise TypeError('name must be string.')


def _validate_dtype(dtype):
    if not isinstance(dtype, str):
        raise TypeError('dtype must be string.')
    if not dtype in _ACCEPTED_DTYPES:
        raise ValueError(f'dtype must be one of the accepted dtypes: {_ACCEPTED_DTYPES}')


def _validate_enum(enum):
    if not isinstance(enum, list):
        raise TypeError('enum must be list.')


def _validate_column_input(name, dtype, enum):
    _validate_name(name)
    _validate_dtype(dtype)
    _validate_enum(enum)


class Column:

    def __init__(self, name: str, dtype: str, enum: List = []):
        _validate_column_input(name, dtype, enum)

        self.name = name
        self.dtype = dtype
        self.enum = enum

    def __repr__(self):
        return f'Column{{name: {self.name}, dtype: {self.dtype}, enum: {self.enum}}}'


def _pandera_data_frame_schema_from_columns(column_dict: Dict) -> pa.DataFrameSchema:
    data_frame_schema_dict = {}
    for name, col in column_dict.items():
        if col.enum:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype],
                pa.Check.isin(col.enum)
            )
        else:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype]
            )
    return pa.DataFrameSchema(data_frame_schema_dict)


class DataFrameSchema:

    def __init__(self, columns: list):
        if len(set([col.name for col in columns])) < len(columns):
            raise ValueError('Columns names must be unique.')
        self.column_dict = {col.name: col for col in columns}
        self._data_frame_schema = _pandera_data_frame_schema_from_columns(self.column_dict)

    def modify_column(self, column_name: str, dtype: str = None, enum: List = None):
        _validate_column_input(column_name, dtype, enum)

        initialized_col_names = [col for col in self.column_dict]
        if column_name not in initialized_col_names:
            raise ValueError(
                f'column name must be one of the initialized column names: {initialized_col_names}'
            )
        new_column = Column(column_name, dtype, enum)
        self.column_dict[column_name] = new_column

    def validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate pandas data frame

        Parameters
        ----------
        df: pandas DataFrame

        Raises:
        -------
        SchemaError
            If df does not conform to the schema, a schema error is raised.

        Returns
        -------
        DataFrame
            Validated data frame
        """
        return self._data_frame_schema.validate(df)

    def get_dtypes(self) -> Dict:
        """

        Returns
        -------
        Dict: Dictionary where keys are the column names and values are the dtypes as strings.
        """
        return {col.name: col.dtype for col in self.column_dict}

    def __repr__(self):
        return f'DataFrameSchema{{columns: {self.column_dict}}}'


def _infer_data_frame_schema_from_df(df: pd.DataFrame) -> DataFrameSchema:
    dtype_dict = df.dtypes.astype(str).to_dict()
    return DataFrameSchema([Column(k, dtype_dict[k]) for k in dtype_dict])
