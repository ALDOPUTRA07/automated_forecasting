import streamlit as st

from arima_model.processing.data_manager import convert_data
from streamlit_app.utils.ui import (
    plot_check_data,
    sidebar,
    sidebar_check_data,
    sidebar_selectbox_forecast,
    tab,
    tab_forecasting_result,
    tab_parameter,
    tab_training_result,
)
from streamlit_app.utils.utils import arima_model


st.header('Automed Forecasting')
st.image('static/automed forecasting-crop.png')
st.write(
    '''This project aims to carry out forecasting automatically by just entering data. 
    The forecasting results will come out in a few seconds. 
    The project uses one of the best forecasting algorithms, namely ARIMA.
    '''
)

sidebar()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    option = sidebar_check_data()

    if option:
        time = option[0]
        if st.sidebar.button("Check data", type="primary"):
            data = convert_data(uploaded_file, time=time)
            plot_check_data(data, option)

    st.sidebar.divider()

    forecast, forecast_point = sidebar_selectbox_forecast()

    if forecast and forecast_point:
        time = forecast[0]

        if st.sidebar.button("Forecasting data", type="primary"):

            (
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
            ) = arima_model(uploaded_file, time, forecast_point)

            tab1, tab2, tab3 = tab()

            with tab1:
                tab_forecasting_result(dataframe_plot_result, dataframe_forecast_result)

            with tab2:
                tab_training_result(dataframe_plot, dataframe_results, dataframe_test)

            with tab3:
                tab_parameter(
                    result_stationery,
                    best_diff,
                    dataframe_parameter,
                    best_parameter,
                    full_data,
                )
