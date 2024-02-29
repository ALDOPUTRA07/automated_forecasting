# Automed Forecasting

<br />
<div align="center">
  <a href="">
    <img src="static/automed forecasting.png">
  </a>
</div>

<p></p>

<!-- TABLE OF CONTENTS -->
<details>
  <p>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#background">Background</a></li>
    <li><a href="#business-values">Business Values</a></li>
    <li><a href="#how-to-works">How to Works</a></li>
    <li><a href="#limitations-model">Limitations Model</a></li>
    <li><a href="#next-step">Next Step</a></li>
    <li><a href="#mlops-concepts">MLOPS Concepts</a></li>
    <li><a href="#aiml-canvas">AI/ML Canvas</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
  </p>
</details>


<p></p>

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to carry out forecasting automatically by just entering data. The forecasting results will come out in a few seconds. The project uses one of the best forecasting algorithms, namely ARIMA.

<video controls muted>
  <source src="static/Tutorial.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
<p></p>


https://github.com/ALDOPUTRA07/automated_forecasting/assets/123882693/0fd2954f-86ef-451f-b73b-e4e8aa147a4d



### Built With

These are list any major frameworks/libraries used to make the project.

* [![Pytorch][Pytorch]][Pytorch-url]
* [![Streamlit][Streamlit]][Streamlit-url]

## Background

Time Series Forecasting is critical in areas such as finance, manufacturing, health, weather studies, and social sciences. These fields could rely on the predictions of future needs, such as sales numbers. The predictions could assist in fine-tuning their stock and boost the efficiency of their supply chain. Hence, precise forecasting is necessary for businesses to adapt to market shifts and stay ahead of competitors.

End users needing predictions need to learn how to forecast data, so this project will make it easier for end users to use forecasting for any data.

## Business Values
- **Easy to use**, by just entering the data, the forecasting results will come out immediately.
- **Cost reduction** forecasting also helps companies reduce costs by providing companies the foresight not to order more stock than necessary to fulfill customer orders.
- **Inventory management and reduction**, If a manufacturer can better understand and predict demand or orders for certain products, they can more effectively work with suppliers to achieve optimal inventory levels and reduce the likelihood of part overages or shortages.

## How to Works
Like in the video, the user only needs to enter data (with 2 columns, date and target column). Users can choose data based on weekly or monthly. Finally, the user chooses how many forecasting points will be predicted.

What happens in the prediction calculations is that this project uses the ARIMA model to make predictions. The following is a brief explanation of how the model works.

1. **Feature engineering**: The data will be changed to weekly or monthly.
2. **Stationarity Check**: Before applying ARIMA, the time series must be checked to ensure they are stationary. This uses statistical tests or visual inspection of the series plot. If the series is non-stationary, differencing (integration) is required to achieve stationarity.
    - Differencing (integration) is carried out until differencing = 3; if the data is still not stationary, it will return the message "Data is not stationary.".
3. **Autocorrelation and Partial Autocorrelation Analysis**: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots are used to identify the values of p and q. These plots help determine the correlation between the time series and its lagged values, indicating the appropriate number of AR and MA terms.
    - After getting the differencing values for stationary data, the model will calculate **p** and **q** values with a high correlation (exceeding the confidence interval).
    - Then, we will train the model with a combination of p, d, and q values.
    - The combination with the best value (evaluation using BIC, AIC, MSE RSE) will be used to predict data based on the number of forecasting points.
4. The model will predict based on the best parameters and the forecasting points.

## Limitations Model
- ARIMA can be sensitive to outliers and extreme values, potentially leading to inaccurate forecasts.
- ARIMA does not perform well with complex data patterns; it assumes that the relationship between future values and past values is linear or near-linear
- ARIMA is unsuitable for very short or long series as they may need more information or become unstable over time.

## Next Step 
This model does not yet include seasonal ARIMA (SARIMA). In the future, we can incorporate this model to improve project quality.

## MLOPS Concepts
<p align="center">
  <img src = "static/MLOPS.png">
</p>

- **Versioning**. The project includes automated versioning using semantic release Python.
- **CI/CD**. The project uses the CI/CD concept with GitHub action.
- **Testing**. Testing includes data, model, and code testing.
- **Reproducibility**.
- **Monitoring**. For the next step, the project can monitor forecasting results. We can use evidently for tool monitoring. The monitoring includes data drift, target drift, etc.

## AI/ML Canvas
Link AI/Canvas for this project. [LINK](https://github.com/ALDOPUTRA07/automated_forecasting/blob/main/static/AI_ML%20Canvas%20Automated%20Forecasting.pdf)
<p align="center">
  <img src = "static/AI_ML Canvas Automated Forecasting.png">
</p>


<!-- GETTING STARTED -->
## Getting Started
This is a tutorial for running a project locally.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ALDOPUTRA07/automated_forecasting
   ```
2. Change to the project directory
   ```sh
   cd automated_forecasting
   ```
3. Setting up programming environment to run the project
   ```sh
   poetry shell
   ```
4. Install the dependencies
   ```sh
   poetry install
   ```
5. Running the project (Without 🐳 Docker)
   ```sh
   streamlit run streamlit_app/app.py
   ```

## License
MIT

<p align="right">(<a href="#automed-forecasting">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
