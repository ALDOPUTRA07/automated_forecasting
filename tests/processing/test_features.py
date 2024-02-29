import pandas as pd

from arima_model.processing.data_manager import convert_data, split_data
from arima_model.processing.features import (
    check_stationery,
    determine_pdq_values,
    result_parameter,
)


def test_check_stationery(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    result_stationery, list_diff_stationery, best_diff = check_stationery(dataset)

    # assert
    assert len(result_stationery) == 4
    assert len(list_diff_stationery) <= 4
    assert best_diff in [1, 2, 3, 4]


def test_determine_pdq_values(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    p_d_q = determine_pdq_values(dataset, d=3)

    # assert
    assert len(p_d_q) > 0


def test_result_parameter(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    _, train_data, _ = split_data(dataset, d=3, perc_test=5)
    p_d_q = determine_pdq_values(dataset, d=3)
    dataframe_parameter, best_parameter = result_parameter(train_data, p_d_q)

    # assert
    assert isinstance(dataframe_parameter, pd.DataFrame)
    assert len(best_parameter) > 0
