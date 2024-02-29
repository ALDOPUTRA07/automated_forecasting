"""
visualization.py
====================================
This script uses for feature engineering such as calculate parameter.
"""

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from arima_model.processing.data_manager import data_with_diff


def plot_stationery(data, d):
    """Plotting data with differencing method.

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    d : int
        Differencing (integration).

    Returns
    -------
    plot

    d : int
        Differencing (integration) values.

    """
    # Load data with differencing
    dataframe = data_with_diff(data, d)
    dataframe = dataframe.rename(columns={dataframe.columns.item(): 'values'})

    return dataframe.plot(), print('d values:', d)


def plot_pdq_values(data, d):
    """Plotting PACF and ACF

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    d : int
        Differencing (integration).

    Returns
    -------
    plot

    """
    # Load data with differencing
    dataframe = data_with_diff(data, d)

    # Plotting ACF and PACF
    plot_acf(dataframe, lags=15)
    plot_pacf(dataframe, lags=15)
    plt.show()
