from typing import List
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

    def __init__(self, name: str, dtype: str, enum:List = []):

        _validate_column_input(name, dtype, enum)

        self.name = name
        self.dtype = dtype
        self.enum = enum

    def update_enum(self, enum):
        _validate_enum(enum)
        self.enum = enum

    def __repr__(self):
        return f'Column{{name: {self.name}, dtype: {self.dtype}, enum: {self.enum}}}'


def _pandera_data_frame_schema_from_columns(columns:List) -> pa.DataFrameSchema:

    data_frame_schema_dict = {}
    for col in columns:
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
        self.columns = columns
        self._data_frame_schema = _pandera_data_frame_schema_from_columns(columns)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def __repr__(self):
        return f'DataFrameSchema{{columns: {self.columns}}}'


def _infer_data_frame_schema_from_df(df: pd.DataFrame) -> DataFrameSchema:
    dtype_dict = df.dtypes.astype(str).to_dict()
    return DataFrameSchema([Column(k, dtype_dict[k]) for k in dtype_dict])