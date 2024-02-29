"""
train_pipeline.py
====================================
This script uses to training and testing ARIMA model.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def train_test_arima(data, p_d_q, test):
    """Returing result of training and testing ARIMA model..

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    p_d_q : list
        Best p,d,q values.

    test : int
        Length of testing data.

    Returns
    -------
    dataframe_results : dataframe
        Dataframe results of training and testing model.

    dataframe_plot : dataframe
        Dataframe for plotting results of training and testing model.

    dataframe_test : dataframe
        Dataframe results of testing model.

    """
    # Load data
    data = data['target']
    len_data = len(data)

    # Splitting data to train and test data
    data_train = data.iloc[: (len_data - test)]
    data_test = data.iloc[(len_data - test) :]

    # Defining evaluation metrics
    order = []
    mae_train, mae_test = [], []
    mse_train, mse_test = [], []
    aic_train, bic_train = [], []

    # Training model
    model = ARIMA(data_train.values, order=p_d_q)
    results = model.fit()

    # Getting evaluation metrics
    order.append(p_d_q)
    mae_train.append(results.mae)
    mse_train.append(results.mse)
    aic_train.append(results.aic)
    bic_train.append(results.bic)

    # Forecasting data based on length data test
    forecast_test = results.forecast(len(data_test))

    # Calculating residuals test
    residuals_test = data_test - forecast_test

    # Get average residuals test
    mae = np.mean(np.abs(residuals_test))
    mse = np.mean(residuals_test**2)

    mae_test.append(mae)
    mse_test.append(mse)

    # Making dataframe results
    dataframe_results = pd.DataFrame(
        list(
            zip(order, mae_train, mse_train, aic_train, bic_train, mae_test, mse_test)
        ),
        columns=[
            'p_d_q values',
            'mae_train',
            'mse_train',
            'aic_train',
            'bic_train',
            'mae_test',
            'mse_test',
        ],
    )

    # Making dataframe for plotting results of training and testing model
    dataframe_plot = pd.DataFrame(
        {
            'values': data.values,
            'forecast_train': list(results.fittedvalues) + [None] * (len(data_test)),
            'predict': [None] * (len(data_train)) + list(forecast_test),
        },
        index=data.index,
    )

    # Making dataframe results of testing model
    dataframe_test = pd.DataFrame(
        {'data_actual_test': data_test.values, 'predict': forecast_test},
        index=data_test.index,
    )

    blankIndex = [''] * len(dataframe_results)
    dataframe_results.index = blankIndex

    return dataframe_results, dataframe_plot, dataframe_test
