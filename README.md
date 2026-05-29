# Machine Learning API Deployment for Bitcoin Price Prediction

This project presents a **FastAPI** application designed to predict the price of Bitcoin using a previously trained Machine Learning model.

The API automatically retrieves recent Bitcoin historical data, calculates multiple technical indicators, applies the required preprocessing steps, and returns an estimated price prediction for the next day.

## Project Objective

The main objective of this project is to demonstrate the development and deployment of a Machine Learning API capable of:

* Collecting historical Bitcoin price data;
* Calculating technical financial indicators;
* Applying data scaling using a saved scaler;
* Loading a trained Machine Learning model;
* Returning a Bitcoin price prediction through a REST API.

## Technologies Used

* Python
* FastAPI
* Uvicorn
* Pandas
* NumPy
* Scikit-learn
* Joblib
* YFinance
* Requests

## Project Structure

```text
.
├── app.py                  # Main API script
├── cliente.py              # Client application to consume the API
├── indicadores.py          # Functions for calculating technical indicators
├── modelo_dsa.joblib       # Trained Machine Learning model
├── scaler_dsa.bin          # Scaler used for data normalization
├── requirements.txt        # Project dependencies
├── LEIAME.txt              # Original execution instructions
└── README.md               # Project documentation
```

## How the Project Works

The application follows this workflow:

1. The user sends a request to the `/predict` endpoint;
2. The API downloads the latest Bitcoin historical data using `yfinance`;
3. Several technical indicators are calculated, such as RSI, MACD, Bollinger Bands, ADX, ATR, OBV, and others;
4. Missing values are handled;
5. The saved scaler is loaded and used to normalize the input data;
6. The trained Machine Learning model is loaded;
7. The API returns the predicted Bitcoin price for the next day in JSON format.

## Technical Indicators

The project calculates a wide range of financial indicators, including:

* Williams %R
* Rate of Change
* RSI
* MACD
* Bollinger Bands
* Ichimoku Cloud
* Exponential Moving Averages
* ADX
* Donchian Channel
* ALMA
* True Strength Index
* Z-Score
* Log Return
* Vortex Indicator
* Aroon Indicator
* ATR
* Keltner Channels
* Chaikin Volatility
* OBV
* Chaikin Money Flow
* Volume Price Trend
* Ease of Movement

These indicators are used to transform historical price and volume data into explanatory variables for the Machine Learning model.

## Installation

Create a Conda virtual environment:

```bash
conda create --name dsaengsoftp2 python=3.12
```

Activate the environment:

```bash
conda activate dsaengsoftp2
```

Install `pip` and the project dependencies:

```bash
conda install pip
pip install -r requirements.txt
```

## Running the API

From the project folder, run:

```bash
python app.py
```

The API will be available at:

```text
http://localhost:3000
```

The automatic API documentation can be accessed at:

```text
http://localhost:3000/docs
```

## Making a Prediction

With the API running, open another terminal, activate the virtual environment, and execute the client script:

```bash
python cliente.py
```

The client sends a request to the `/predict` endpoint using the following payload:

```json
{
  "Model": "Machine Learning"
}
```

The API returns a response similar to:

```json
{
  "Modelo": "Machine Learning",
  "Último Preço": 104500.25,
  "Previsão Para o Próximo Dia": 105230.80
}
```

## Main Endpoint

### POST `/predict`

This endpoint is responsible for generating the Bitcoin price prediction.

#### Request Example

```json
{
  "Model": "Machine Learning"
}
```

#### Response Example

```json
{
  "Modelo": "Machine Learning",
  "Último Preço": 104500.25,
  "Previsão Para o Próximo Dia": 105230.80
}
```

## Notes

* The trained model must be available as `modelo_dsa.joblib`.
* The scaler used during preprocessing must be available as `scaler_dsa.bin`.
* The prediction depends on the most recent market data retrieved from `yfinance`.
* This project was developed for educational purposes and demonstrates the process of building, deploying, and consuming a Machine Learning API.

## Author

Project developed as a practical study on building and deploying APIs for Machine Learning models.
