from typing import List, Dict, Union
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


class Interval:
    """Class for holding intervals."""
    def __init__(self, start_value: float, end_value: float):

        if end_value <= start_value:
            raise ValueError('end_value must be larger than start_value.')

        self.start_value = start_value
        self.end_value = end_value

    def __repr__(self):
        return f'Interval{{start_value: {self.start_value}, end_value: {self.end_value}}}'


class Column:
    """Class for holding schema information for individual columns."""
    def __init__(
            self, name: str,
            dtype: str,
            enum: Union[List, None] = None,
            interval: Union[Interval, None] = None
    ):
        _validate_column_input(name, dtype, enum, interval)

        self.name = name
        self.dtype = dtype
        self.enum = enum
        self.interval = interval

    def __repr__(self):
        return f'Column{{name: {self.name}, dtype: {self.dtype}, enum: {self.enum}, interval: {self.interval}}}'


def _pandera_data_frame_schema_from_column_dict(column_dict: Dict) -> pa.DataFrameSchema:
    data_frame_schema_dict = {}
    for name, col in column_dict.items():

        checks = []
        if col.enum:
            checks.append(pa.Check.isin(col.enum))
        if col.interval:
            checks.append(pa.Check(lambda s: s.between(col.interval.start_value, col.interval.end_value)))

        if checks:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype],
                checks
            )
        else:
            data_frame_schema_dict[col.name] = pa.Column(
                _dtype_to_pandera_map[col.dtype]
            )
    return pa.DataFrameSchema(data_frame_schema_dict)


class DataFrameSchema:

    def __init__(self, columns: List[Column]):
        if len(set([col.name for col in columns])) < len(columns):
            raise ValueError('Columns names must be unique.')
        self.column_dict = {col.name: col for col in columns}
        self._data_frame_schema = _pandera_data_frame_schema_from_column_dict(self.column_dict)

    def modify_column(
            self,
            column_name: str,
            dtype: Union[str, None] = None,
            enum: Union[List[str], None] = None,
            interval: Union[Interval, None] = None,
    ):
        _validate_name(column_name)
        if dtype:
            _validate_dtype(dtype)
        if enum:
            _validate_enum(enum)
        if interval:
            _validate_interval(interval)

        initialized_col_names = [col for col in self.column_dict]
        if column_name not in initialized_col_names:
            raise ValueError(
                f'column name must be one of the initialized column names: {initialized_col_names}'
            )

        new_column = Column(
            column_name,
            dtype or self.column_dict[column_name].dtype,
            enum or self.column_dict[column_name].enum,
            interval or self.column_dict[column_name].interval)
        self.column_dict[column_name] = new_column
        self._data_frame_schema = _pandera_data_frame_schema_from_column_dict(self.column_dict)

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
        return {col.name: col.dtype for _, col in self.column_dict.items()}

    def __repr__(self):
        return f'DataFrameSchema{{columns: {self.column_dict}}}'


def _infer_data_frame_schema_from_df(df: pd.DataFrame) -> DataFrameSchema:
    dtype_dict = df.dtypes.astype(str).to_dict()
    return DataFrameSchema([Column(k, dtype_dict[k]) for k in dtype_dict])


def _get_enums_from_data_frame(df: pd.DataFrame, enum_columns: List[str]) -> Dict:
    enum_dict = {col: list(df[col].unique()) for col in enum_columns}
    return enum_dict


def _get_intervals_from_data_frame(df: pd.DataFrame, interval_columns: List[str]) -> Dict:
    interval_dict = {col: Interval(df[col].min(), df[col].max()) for col in interval_columns}
    return interval_dict


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


def _validate_interval(interval):
    if not isinstance(interval, Interval):
        raise TypeError('interval must be pandas Interval.')


def _validate_column_input(name, dtype, enum, interval):
    _validate_name(name)
    _validate_dtype(dtype)
    if enum:
        _validate_enum(enum)
    if interval:
        _validate_interval(interval)