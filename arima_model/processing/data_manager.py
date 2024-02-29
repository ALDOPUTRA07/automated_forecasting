"""
data_manager.py
====================================
This script uses to manage data such as loading data, splitting data, 
checking data, and converting data.
"""


from numbers import Number
import pandas as pd


def load_check_data(data) -> pd.DataFrame:
    """Checking data.

    Data must have 2 columns (date and target),
    data is in a CSV file, the target column is numeric,
    data has a date column.

    Parameters
    ----------
    data : CSV file
        Data file for forecasting.

    Returns
    -------
    Dataframe

    """
    # Check csv file
    try:
        dataframe = pd.read_csv(data)
    except:
        raise ValueError('Data is not a csv file')

    # Check data have 2 columns
    if len(dataframe.columns) != 2:
        raise ValueError("Data must have 2 columns")

    # Check data has numeric column
    if Number in dataframe.dtypes.to_list() is True:
        raise ValueError("Data must has a integer column")

    # Check data has date column
    # Convert and rename columns
    columns = dataframe.columns
    try:
        dataframe[columns[0]] = pd.to_datetime(dataframe[columns[0]])
        dataframe.rename(columns={columns[0]: 'date'}, inplace=True)
        dataframe.rename(columns={columns[1]: 'target'}, inplace=True)
    except:
        try:
            dataframe[columns[1]] = pd.to_datetime(dataframe[columns[1]])
            dataframe.rename(columns={columns[1]: 'date'}, inplace=True)
            dataframe.rename(columns={columns[0]: 'target'}, inplace=True)
        except:
            raise ValueError('Data must has a date column')

    return dataframe


def convert_data(data: pd.DataFrame, time: str) -> pd.DataFrame:
    """Convert data to weekly or monthly.


    Parameters
    ----------
    data : Dataframe
        Data source for forecasting.

    time : str
        'W' is option to forecasting weekly.
        'M' is option to forecasting monthly.

    Returns
    -------
    Dataframe

    """
    # Load data
    dataframe = load_check_data(data)

    # Convert to weekly or monthly
    if time == 'W':
        data_time_series = dataframe.groupby(
            dataframe['date'].dt.to_period('W').dt.start_time
        ).sum('target')

    if time == 'M':
        data_time_series = dataframe.groupby(
            dataframe['date'].dt.to_period('M').dt.start_time
        ).sum('target')

    return data_time_series['target'].to_frame()


def data_with_diff(data: pd.DataFrame, d) -> pd.DataFrame:
    """Differencing (integration) data.

    Parameters
    ----------
    data : Dataframe
        Data source for forecasting.

    d : int
        Differencing (integration).

    Returns
    -------
    Dataframe

    """
    # looping for multiple differencing (integration)
    for x in range(d, d + 1):
        dataframe = pd.eval('data' + '.diff()' * x + '.dropna()')

    # Dropping null values after differencing
    df = dataframe.dropna()

    return df


def split_data(data: pd.DataFrame, d: float, perc_test: float) -> pd.DataFrame:
    """Split data to training and data testing

    Parameters
    ----------
    data : Dataframe
        Data source for forecasting.

    d : int
        Differencing (integration).

    perc_test : float
        Percentage data for data testing.

    Returns
    -------
    full_data : Dataframe
        Data with differencing (integration).
    
    train_data : Dataframe
        Data with differencing (integration) for training model.
    
    test_data : Dataframe
        Data with differencing (integration) for testing model.

    """
    # differencing (integration) data
    dataframe = data_with_diff(data, d)

    # Test data is only 5%
    # Splitting to training and testing data.
    len_data = len(dataframe)
    len_test = round((perc_test / 100) * len_data)

    train_data = dataframe.iloc[: (len_data - len_test)]
    test_data = dataframe.iloc[(len_data - len_test) :]

    # full_data with differencing (integration)
    full_data = dataframe

    return full_data, train_data, test_data
