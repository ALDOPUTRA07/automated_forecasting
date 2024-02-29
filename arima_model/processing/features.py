"""
features.py
====================================
This script uses for feature engineering such as calculate parameter.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

from arima_model.processing.data_manager import data_with_diff


def check_stationery(data):
    """Checking stationey data.

    This function will check whether the data is stationary. 
    otherwise the differencing method will be executed.

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    Returns
    -------
    result_stationery : dataframe
        Results of differencing method.

    list_diff_stationery : list
        List of differencing values (sorting by statistics values).

    best_diff : int
        Best differencing values based on statistics values.

    """

    # differencing method
    data_frame = data.rename(columns={'target': 'diff_0'})
    data_frame['diff_1'] = data_frame['diff_0'].diff().fillna(0)
    data_frame['diff_2'] = data_frame['diff_0'].diff().diff().fillna(0)
    data_frame['diff_3'] = data_frame['diff_0'].diff().diff().diff().fillna(0)

    # Calculate statistics values of differencing method
    diff, test_statistics, p = [], [], []
    critical_value_1_percent, critical_value_5_percent, critical_value_10_percent = (
        [],
        [],
        [],
    )

    for x, column in enumerate(data_frame):
        res = adfuller(data_frame[column])

        diff.append(x)
        test_statistics.append(res[0])
        p.append(res[1])
        critical_value_1_percent.append(res[4].get('1%'))
        critical_value_5_percent.append(res[4].get('5%'))
        critical_value_10_percent.append(res[4].get('10%'))

    # Making dataframe from Calculate statistics values of differencing method
    df = pd.DataFrame(
        list(
            zip(
                data_frame.columns,
                diff,
                test_statistics,
                p,
                critical_value_1_percent,
                critical_value_5_percent,
                critical_value_10_percent,
            )
        ),
        columns=[
            'Diff',
            'diff_values',
            'test_statistics',
            'p_values',
            'critical_value_1_percent',
            'critical_value_5_percent',
            'critical_value_10_percent',
        ],
    )

    df['is_stationery'] = np.where(
        (df['p_values'] < 0.05)
        & (df['test_statistics'] < df['critical_value_1_percent'])
        & (df['test_statistics'] < df['critical_value_5_percent'])
        & (df['test_statistics'] < df['critical_value_10_percent']),
        True,
        False,
    )

    # Sort values by differencing values
    result_stationery = df.sort_values(
        ['diff_values']
    )

    list_diff_stationery = list(
        result_stationery[result_stationery['is_stationery'] == True].diff_values
    )

    # if the data is still not stationary, even though it has been differentiated 3 times. 
    # then it will return the following message.
    if len(list_diff_stationery) == 0:
        raise ValueError('Data is non-stationery')

    # Getting the best differencing value
    best_diff = list_diff_stationery[0]

    return result_stationery, list_diff_stationery, best_diff


def determine_pdq_values(data, d):
    """determining p,d,q values of ARIMA model.

    After getting the differencing values for stationary data, 
    the model will calculate P values and Q values with a high correlation 
    (exceeding the confidence interval).

    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    d : int
        Differencing (integration).

    Returns
    -------
    p_d_q : list
        list combiantions of p,d,q values.

    """
    # Differencing data
    dataframe = data_with_diff(data, d)

    # Calculate PACF
    pacf_values, confint_pacf = pacf(
        dataframe.values, nlags=15, alpha=0.05, method='ywm'
    )

    # Calculate confident intervals
    lower_bound_pacf = confint_pacf[:, 0] - pacf_values
    upper_bound_pacf = confint_pacf[:, 1] - pacf_values

    # Calculate PACF
    acf_values, confint_acf = acf(
        dataframe.values,
        nlags=15,
        alpha=0.05,
        fft=False,
        bartlett_confint=True,
        adjusted=False,
        missing="none",
    )

    # Calculate confident intervals
    lower_bound_acf = confint_acf[:, 0] - acf_values
    upper_bound_acf = confint_acf[:, 1] - acf_values

    idx_pacf = []
    values_pacf_abs = []
    for idx, (values, lower_bound, upper_bound) in enumerate(
        zip(pacf_values.tolist(), lower_bound_pacf.tolist(), upper_bound_pacf.tolist())
    ):
        if (values > upper_bound or values < lower_bound) and idx != 0:
            idx_pacf.append(idx)
            values_pacf_abs.append(abs(values))

    idx_acf = []
    values_acf_abs = []
    for idx, (values, lower_bound, upper_bound) in enumerate(
        zip(acf_values.tolist(), lower_bound_acf.tolist(), upper_bound_acf.tolist())
    ):
        if (values > upper_bound or values < lower_bound) and idx != 0:
            idx_acf.append(idx)
            values_acf_abs.append(abs(values))

    # If p and q is greater then 3, we will get the best 3
    lag_pacf = pd.DataFrame({'lags': idx_pacf, 'values': values_pacf_abs}).sort_values(
        'values', ascending=False
    )
    lag_pacf = lag_pacf.values[:3, 0].tolist()

    lag_acf = pd.DataFrame({'lags': idx_acf, 'values': values_acf_abs}).sort_values(
        'values', ascending=False
    )
    lag_acf = lag_acf.values[:3, 0].tolist()

    # collecting combinations of p,d,q values
    p_d_q = []
    for p in lag_pacf:
        for d in range(d, d + 1):
            for q in lag_acf:
                a = [int(p), d, int(q)]
                p_d_q.append(a)

    return p_d_q


def result_parameter(data, p_d_q):
    """calculating statistics values of combinations p,d,q values

    we will train the model with a combination of p, d, and q values.
    The combination with the best value (evaluation using BIC, AIC, MSE RSE) 
    will be used to predict data based on the number of forecasting points.


    Parameters
    ----------
    data : dataframe
        Data file for forecasting.

    p_d_q : int
        List combinations of p,d,q values.

    Returns
    -------
    dataframe

    best_parameter : list
        Best combination of p,d,q values
    """

    # Load data
    data = data['target']
    order, mae, mse, aic, bic = [], [], [], [], []

    # Training model based on list combinations of p,d,q values
    # Getting statistics values from the training
    for x in p_d_q:
        model = ARIMA(data.values, order=x)
        results = model.fit()

        order.append(x)
        mae.append(results.mae)
        mse.append(results.mse)
        aic.append(results.aic)
        bic.append(results.bic)

    # Making dataframe
    dataframe = pd.DataFrame(
        list(zip(order, mae, mse, aic, bic)),
        columns=['order', 'mae', 'mse', 'aic', 'bic'],
    )
    dataframe = dataframe.sort_values(by=['mae'])
    
    # Getting best combination of p,d,q values
    best_parameter = dataframe.iloc[0, 0]

    return dataframe.reset_index(drop=True), best_parameter
