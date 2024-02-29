import pandas as pd

from arima_model.predict import result_forecast
from arima_model.processing.data_manager import convert_data


def test_result_forecast(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    dataframe_plot_result, dataframe_forecast_result = result_forecast(
        dataset, [1, 1, 1], 10
    )

    # assert
    assert isinstance(dataframe_plot_result, pd.DataFrame)
    assert isinstance(dataframe_forecast_result, pd.DataFrame)
