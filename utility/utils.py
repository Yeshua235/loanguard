import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_distribution_frame(data: pd.DataFrame) -> pd.DataFrame:
    '''
    function for inspecting the data distribution of the target classes

    :param data: The target matrix
    :type data: pd.DataFrame
    :return: A dataframe showing the value counts for each class, and its percent of the total target column
    :rtype: DataFrame
    '''
    result_frame =  pd.concat(
                            (pd.Series(data.value_counts(), name='value counts'),
                            pd.Series([f'{round(val/data.shape[0]*100)}%' for val in data.value_counts()], name='percent of total')),
    axis=1
                    )
    return result_frame


def columns_with_null(data: pd.DataFrame, option:str='') -> pd.DataFrame:
    '''
    Function for displaying columns containing null values in a dataframe in a more informative manner.

    :param data: The Data Matrix
    :type data: pd.DataFrame
    :param option: column dtype to return. Numeric columns [option="numeric"], Non-numeric columns [option="non-numeric] or both (default).
    :return: A DataFrame showing columns with null values, their datatype, the amount of nulls present in count and in percentage of the total
    :rtype: DataFrame
    '''
    cols = data.columns

    float_bad_cols = {}
    object_bad_cols = {}

    for col in cols:
        nulls = sum(data[col].isnull())
        data_type = data[col].dtype
        if nulls:
            if data_type in ['float64', 'int64']:
                float_bad_cols[col] = nulls
            else:
                object_bad_cols[col] = nulls

    num_bad_cols = [(col, data[col].dtype, round(float_bad_cols[col]/data.shape[0] * 100), float_bad_cols[col]) for col  in float_bad_cols.keys()]
    str_bad_cols = [(col, data[col].dtype, round(object_bad_cols[col]/data.shape[0] * 100), object_bad_cols[col]) for col in object_bad_cols.keys()]

    num_bad_frame = pd.DataFrame(num_bad_cols, columns=['column name', 'column dtype', 'percent of total', 'number of nulls'])
    str_bad_frame = pd.DataFrame(str_bad_cols, columns=['column name', 'column dtype', 'percent of total', 'number of nulls'])

    if option == 'numeric':
        return num_bad_frame
    elif option == 'non-numeric':
        return str_bad_frame
    else:
        null_col_frames = pd.concat((num_bad_frame, str_bad_frame), axis=0)
        return null_col_frames

