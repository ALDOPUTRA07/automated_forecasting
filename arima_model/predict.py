"""
predict.py
====================================
This script uses for feature engineering such as calculate parameter.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def result_forecast(data, p_d_q, forecast_point):
    """Returing result of forecasting.

    The model will predict based on the best parameters and several forecasting points

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    p_d_q : list
        Best p,d,q values.

    forecast_point : int
        Forecasting point.

    Returns
    -------
    dataframe_plot : dataframe
        dataframe for plotting results of forecasting.

    dataframe_forecast : dataframe
        results of forecasting.

    """
    # Load data
    data = data['target']

    # Training model
    model = ARIMA(data.values, order=p_d_q)
    results = model.fit()

    # Forecasting data based on numbers of forecast point
    forecast = results.forecast(forecast_point)

    index_forecast = list(
        pd.to_datetime(
            (
                pd.date_range(
                    start=data.index[-1], periods=(forecast_point + 1), freq='MS'
                )
            ).date
        )
    )

    # Making dataframe for plotting results of forecasting.
    dataframe_plot = pd.DataFrame(
        {
            'values': list(data.values) + [None] * len(forecast),
            'predict': [None] * (len(data)) + list(forecast),
        },
        index=list(data.index) + index_forecast[1:],
    )

    # Making results of forecasting.
    dataframe_forecast = pd.DataFrame({'predict': forecast}, index=index_forecast[1:])

    return dataframe_plot, dataframe_forecast
