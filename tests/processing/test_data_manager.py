import pandas as pd

from arima_model.processing.data_manager import (
    convert_data,
    data_with_diff,
    load_check_data,
    split_data,
)


def test_load_check_data(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = load_check_data(file_dataset)

    # assert
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape[1] == 2


def test_convert_data(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    timedelta = dataset.index[1] - dataset.index[0]

    # assert
    assert isinstance(dataset, pd.DataFrame)
    assert timedelta.days == 7


def test_data_with_diff(dataname):
    # arrange
    file_dataset = dataname

    # act
    data_convert_data = convert_data(file_dataset, 'W')
    data_data_with_diff = data_with_diff(data_convert_data, d=3)

    # assert
    assert data_data_with_diff.isna().sum().target == 0


def test_split_data(dataname):
    # arrange
    file_dataset = dataname

    # act
    dataset = convert_data(file_dataset, 'W')
    full_data, train_data, test_data = split_data(dataset, d=3, perc_test=5)

    # assert
    assert len(train_data) + len(test_data) == len(full_data)
    assert len(test_data) == round(0.05 * len(full_data))
