import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def sidebar() -> None:
    with st.sidebar:
        st.write("Automated Forecasting")


def sidebar_check_data():
    option = st.sidebar.selectbox(
        "Checking data",
        ("Weekly", "Monthly"),
        index=None,
        placeholder="Select contact method...",
    )
    return option


def plot_check_data(data: pd.DataFrame, option) -> None:
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(title="Data " + str(option), xlabel=option[:-2], ylabel='Values')
    st.pyplot(fig)

    st.write(
        "Data from ", str(data.index[0].date()), " to ", str(data.index[-1].date())
    )
    st.write(data)


def sidebar_selectbox_forecast():
    forecast = st.sidebar.selectbox(
        "Choosing Weekly or Monthly",
        ("Weekly", "Monthly"),
        index=None,
        placeholder="Select contact method...",
    )

    forecast_point = st.sidebar.selectbox(
        "Forecasting point",
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        index=None,
        placeholder="Select contact method...",
    )

    return forecast, forecast_point


def tab():
    tab1, tab2, tab3 = st.tabs(["Forecasting Result", "Training Result", "Parameter"])
    return tab1, tab2, tab3


def tab_forecasting_result(dataframe_plot_result, dataframe_forecast_result):
    st.header("Forecasting Result", divider='rainbow')
    fig, ax = plt.subplots()
    ax.plot(dataframe_plot_result)
    ax.set(title="Forecast Result", xlabel="Date", ylabel='Values')
    st.pyplot(fig)

    st.sidebar.divider()

    st.subheader("Data Prediction")
    st.write(dataframe_forecast_result)


def tab_training_result(dataframe_plot, dataframe_results, dataframe_test):
    st.header("Training Result", divider='rainbow')
    fig, ax = plt.subplots()
    ax.plot(dataframe_plot)
    ax.set(title="Training Result", xlabel="Date", ylabel='Values')
    st.pyplot(fig)

    st.subheader("Training and Testing Result")
    st.markdown(
        """ **p_d_q values** is the best values of parameter p, d, and q.
        **Another columns** are results of training forecast."""
    )

    st.write(dataframe_results)

    st.sidebar.divider()

    st.subheader("Prediction Test Data")
    st.markdown("""Testing data is :red[5%] from data.""")
    st.write(dataframe_test)


def tab_parameter(
    result_stationery, best_diff, dataframe_parameter, best_parameter, full_data
):
    st.header("Overview Parameter", divider='rainbow')

    st.subheader("Result Stationery")
    st.write("**Diff value** is how many differencing function does to my data.")
    st.write(result_stationery)

    st.sidebar.divider()

    st.subheader("p,d,q Parameter")
    st.write(
        "To be a stationery, data must does  :red[", str(best_diff), "]  differencing methods."
    )
    st.write("This is only get the best 3 of p and q values.")

    st.sidebar.divider()

    st.write("Results of analyzing p,d,q parameter ")
    st.write(dataframe_parameter)

    st.write("Best p,d,q values is  :red[", str(best_parameter), "]")

    st.sidebar.divider()

    st.subheader("PACF and ACF chart")
    st.write("PACF and ACF chart from best p,d,q values (:red[", str(best_parameter), "] )")
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(full_data, lags=15, ax=ax[0])
    plot_pacf(full_data, lags=15, ax=ax[1])
    plt.show()
    st.pyplot(fig)
