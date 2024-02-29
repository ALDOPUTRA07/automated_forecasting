from arima_model.predict import result_forecast
from arima_model.processing.data_manager import convert_data, split_data
from arima_model.processing.features import (
    check_stationery,
    determine_pdq_values,
    result_parameter,
)
from arima_model.train_pipeline import train_test_arima


def arima_model(uploaded_file, time, forecast_point):
    data = convert_data(uploaded_file, time=time)
    result_stationery, list_diff_stationery, best_diff = check_stationery(data)
    p_d_q = determine_pdq_values(data, d=best_diff)
    full_data, train_data, test_data = split_data(data, best_diff, perc_test=5)
    dataframe_parameter, best_parameter = result_parameter(train_data, p_d_q)
    dataframe_results, dataframe_plot, dataframe_test = train_test_arima(
        data, best_parameter, test=len(test_data)
    )
    dataframe_plot_result, dataframe_forecast_result = result_forecast(
        data, best_parameter, forecast_point
    )

    return (
        result_stationery,
        best_diff,
        full_data,
        dataframe_parameter,
        best_parameter,
        dataframe_results,
        dataframe_plot,
        dataframe_test,
        dataframe_plot_result,
        dataframe_forecast_result,
    )
