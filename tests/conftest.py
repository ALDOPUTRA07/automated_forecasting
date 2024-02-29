import pandas as pd
import pytest


@pytest.fixture
def dataset():
    data = pd.read_csv('tests/data/testing_data.csv')

    return data


@pytest.fixture
def dataname():
    filename = 'tests/data/testing_data.csv'

    return filename
