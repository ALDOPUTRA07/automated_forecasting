FROM python:3.9

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /app

COPY ./streamlit_app /app/streamlit_app
COPY ./arima_model /app/arima_model
COPY pyproject.toml /app

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

RUN chown -R ml-api-user:ml-api-user ./

EXPOSE 8001

CMD cd streamlit_app && streamlit run app.py