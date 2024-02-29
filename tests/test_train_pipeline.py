import pandas as pd

from arima_model.processing.data_manager import convert_data
from arima_model.train_pipeline import train_test_arima


def test_train_test_arima(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    dataframe_results, dataframe_plot, dataframe_test = train_test_arima(
        dataset, [1, 1, 1], test=10
    )

    # assert
    assert isinstance(dataframe_results, pd.DataFrame)
    assert len(dataframe_results) > 0
    assert isinstance(dataframe_plot, pd.DataFrame)
    assert isinstance(dataframe_test, pd.DataFrame)
