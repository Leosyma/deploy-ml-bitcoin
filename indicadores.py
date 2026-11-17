# Projeto 2 - Construção e Deploy de API - Machine Learning Para Prever o Preço do Bitcoin
# Módulo de Indicadores

# Imports
import numpy as np
import pandas as pd

# Função de cálculo dos indicadores
def calcula_indicadores(dados) :

    dados = ind_williams_percent_r(dados,14)
    dados = ind_roc(dados,14)
    dados = ind_rsi(dados,7)
    dados = ind_rsi(dados,14)
    dados = ind_rsi(dados,28)
    dados = ind_macd(dados, 8, 21)
    dados = ind_bbands(dados,20)    
    dados = ind_ichimoku_cloud(dados)
    dados = ind_ema(dados, 3)
    dados = ind_ema(dados, 8)
    dados = ind_ema(dados, 15)
    dados = ind_ema(dados, 50)
    dados = ind_ema(dados, 100)
    dados = ind_adx(dados, 14)
    dados = ind_donchian(dados, 10)
    dados = ind_donchian(dados, 20)
    dados = ind_alma(dados, 10)
    dados = ind_tsi(dados, 13, 25)
    dados = ind_zscore(dados, 20)
    dados = ind_log_return(dados, 10)
    dados = ind_log_return(dados, 20)
    dados = ind_vortex(dados, 7)
    dados = ind_aroon(dados, 16)
    dados = ind_ebsw(dados, 14)
    dados = ind_accbands(dados, 20)
    dados = ind_short_run(dados, 14)
    dados = ind_bias(dados, 26)
    dados = ind_ttm_trend(dados, 5, 20)
    dados = ind_percent_return(dados, 10)
    dados = ind_percent_return(dados, 20)
    dados = ind_kurtosis(dados, 5)
    dados = ind_kurtosis(dados, 10)
    dados = ind_kurtosis(dados, 20)
    dados = ind_eri(dados, 13)    
    dados = ind_atr(dados, 14)
    dados = ind_keltner_channels(dados, 20)
    dados = ind_chaikin_volatility(dados, 10)
    dados = ind_stdev(dados, 5)
    dados = ind_stdev(dados, 10)
    dados = ind_stdev(dados, 20)
    dados = ta_vix(dados, 21)    
    dados = ind_obv(dados, 10)
    dados = ind_chaikin_money_flow(dados, 5)
    dados = ind_volume_price_trend(dados, 7)
    dados = ind_accumulation_distribution_line(dados, 3)
    dados = ind_ease_of_movement(dados, 14)
    
    return dados

# Williams %R
def ind_williams_percent_r(dados, window=14):
    highest_high = dados["High"].rolling(window=window).max()
    lowest_low = dados["Low"].rolling(window=window).min()
    dados["Williams_%R{}".format(window)] = -((highest_high - dados["Close"]) / (highest_high - lowest_low)) * 100
    return dados

# Rate of Change
def ind_roc(dados, window=14):
    dados["ROC_{}".format(window)] = (dados["Close"] / dados["Close"].shift(window) - 1) * 100
    return dados

# RSI
def ind_rsi(dados, window=14) : 
    delta = dados["Close"].diff(1)
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    dados["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return dados

# MACD 
def ind_macd(dados, short_window=8, long_window=21, signal_window=9):
    short_ema = dados["Close"].ewm(span = short_window, adjust = False).mean()
    long_ema = dados["Close"].ewm(span = long_window, adjust = False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    dados["MACD_Line"] = macd_line
    dados["Signal_Line"] = signal_line
    dados["MACD_Histogram"] = macd_histogram
    return dados

# Bollinger Bands
def ind_bbands(dados, window=20, num_std_dev=2) :
    dados["midlle_band"] = dados["Close"].rolling(window=window).mean()
    dados["std"] = dados["Close"].rolling(window=window).std()
    dados["upper_band{}".format(window)] = dados["midlle_band"] + (num_std_dev * dados["std"])
    dados["lower_band{}".format(window)] = dados["midlle_band"] - (num_std_dev * dados["std"])
    dados.drop(["std"], axis=1, inplace=True)   
    return dados

# Ichimoku Cloud
def ind_ichimoku_cloud(dados, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    tenkan_sen = (dados["Close"].rolling(window=window_tenkan).max() + dados["Close"].rolling(window=window_tenkan).min()) / 2
    kijun_sen = (dados["Close"].rolling(window=window_kijun).max() + dados["Close"].rolling(window=window_kijun).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (dados["Close"].rolling(window=window_senkou_span_b).max() + dados["Close"].rolling(window=window_senkou_span_b).min()) / 2
    chikou_span = dados["Close"].shift(-window_chikou)
    dados["Tenkan_sen"] = tenkan_sen
    dados["Kijun_sen"] = kijun_sen
    dados["Senkou_Span_A"] = senkou_span_a
    dados["Senkou_Span_B"] = senkou_span_b
    dados["Chikou_Span"] = chikou_span
    return dados

# Moving Average (EMA)
def ind_ema(dados, window=8): 
    dados["ema_{}".format(window)] = dados["Close"].ewm(span=window, adjust=False).mean()
    return dados

# ADX
def ind_adx(dados, window=14): #14
    dados["TR"] = abs(dados["High"] - dados["Low"]).combine_first(abs(dados["High"] - dados["Close"].shift(1))).combine_first(abs(dados["Low"] - dados["Close"].shift(1)))
    dados["DMplus"] = (dados["High"] - dados["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    dados["DMminus"] = (dados["Low"].shift(1) - dados["Low"]).apply(lambda x: x if x > 0 else 0)
    dados["ATR"] = dados["TR"].rolling(window=window).mean()
    dados["DIplus"] = (dados["DMplus"].rolling(window=window).mean() / dados["ATR"]) * 100
    dados["DIminus"] = (dados["DMminus"].rolling(window=window).mean() / dados["ATR"]) * 100
    dados["DX"] = abs(dados["DIplus"] - dados["DIminus"]) / (dados["DIplus"] + dados["DIminus"]) * 100
    dados["ADX_{}".format(window)] = dados["DX"].rolling(window=window).mean()
    dados.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return dados

# Donchian Channel
def ind_donchian(dados, window=10):
    highest_high = dados["Close"].rolling(window=window).max()
    lowest_low = dados["Close"].rolling(window=window).min()
    dados["Donchian_Upper_{}".format(window)] = highest_high
    dados["Donchian_Lower_{}".format(window)] = lowest_low
    return dados

# Arnaud Legoux Moving Average (ALMA)
def ind_alma(dados, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(dados["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    dados["ALMA_{}".format(window)] = alma_values
    return dados

# True Strength Index (TSI)
def ind_tsi(dados, short_period=13, long_period=25):
    price_diff = dados["Close"].diff(1)
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    dados["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return dados

# Z-Score
def ind_zscore(dados, window=20):
    rolling_mean = dados["Close"].rolling(window=window).mean()
    rolling_std = dados["Close"].rolling(window=window).std()
    z_score = (dados["Close"] - rolling_mean) / rolling_std
    dados["Z_Score_{}".format(window)] = z_score
    return dados

# Log Return
def ind_log_return(dados, window=5):
    dados["LogReturn_{}".format(window)] = dados["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return dados

# Vortex Indicator
def ind_vortex(dados, window=7): 
    high_low = dados["High"] - dados["Low"]
    high_close_previous = abs(dados["High"] - dados["Close"].shift(1))
    low_close_previous = abs(dados["Low"] - dados["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    positive_vm = abs(dados["High"].shift(1) - dados["Low"])
    negative_vm = abs(dados["Low"].shift(1) - dados["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    dados["Positive_VI_{}".format(window)] = positive_vi
    dados["Negative_VI_{}".format(window)] = negative_vi
    return dados

# Aroon Indicator
def ind_aroon(dados, window=16):
    high_prices = dados["High"]
    low_prices = dados["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    dados["Aroon_Up_{}".format(window)] = aroon_up
    dados["Aroon_Down_{}".format(window)] = aroon_down
    return dados

# Elder"s Bull Power e Bear Power 
def ind_ebsw(dados, window=14):
    ema = dados["Close"].ewm(span=window, adjust=False).mean()
    bull_power = dados["High"] - ema
    bear_power = dados["Low"] - ema
    dados["Bull_Power_{}".format(window)] = bull_power
    dados["Bear_Power_{}".format(window)] = bear_power
    return dados

# Acceleration Bands
def ind_accbands(dados, window=20, acceleration_factor=0.02):
    sma = dados["Close"].rolling(window=window).mean()
    band_difference = dados["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    dados["Upper_Band_{}".format(window)] = upper_band
    dados["Lower_Band_{}".format(window)] = lower_band
    dados["Middle_Band_{}".format(window)] = sma
    return dados

# Short Run
def ind_short_run(dados, window=14):
    short_run = dados["Close"] - dados["Close"].rolling(window=window).min()
    dados["Short_Run_{}".format(window)] = short_run
    return dados

# Bias
def ind_bias(dados, window=26):
    moving_average = dados["Close"].rolling(window=window).mean()
    bias = ((dados["Close"] - moving_average) / moving_average) * 100
    dados["Bias_{}".format(window)] = bias
    return dados

# TTM Trend
def ind_ttm_trend(dados, short_window=5, long_window=20):
    short_ema = dados["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dados["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    dados["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return dados

# Percent Return
def ind_percent_return(dados, window=1): 
    percent_return = dados["Close"].pct_change().rolling(window=window).mean() * 100
    dados["Percent_Return_{}".format(window)] = percent_return
    return dados

# Kurtosis
def ind_kurtosis(dados, window=20):
    dados["kurtosis_{}".format(window)] = dados["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return dados

# Elder's Force Index (ERI)
def ind_eri(dados, window=13):
    price_change = dados["Close"].diff()
    force_index = price_change * dados["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    dados["ERI_{}".format(window)] = eri
    return dados

# ATR
def ind_atr(dados, window=14):
    dados["High-Low"] = dados["High"] - dados["Low"]
    dados["High-PrevClose"] = abs(dados["High"] - dados["Close"].shift(1))
    dados["Low-PrevClose"] = abs(dados["Low"] - dados["Close"].shift(1))
    dados["TrueRange"] = dados[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    dados["atr_{}".format(window)] = dados["TrueRange"].rolling(window=window, min_periods=1).mean()
    dados.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return dados

# Keltner Channels
def ind_keltner_channels(dados, period=20, multiplier=2):
    dados["TR"] = dados.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    dados["ATR"] = dados["TR"].rolling(window=period).mean()
    dados["Middle Band"] = dados["Close"].rolling(window=period).mean()
    dados["Upper Band"] = dados["Middle Band"] + multiplier * dados["ATR"]
    dados["Lower Band"] = dados["Middle Band"] - multiplier * dados["ATR"]
    return dados

# Chaikin Volatility
def ind_chaikin_volatility(dados, window=10):
    daily_returns = dados["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    dados["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return dados

# Standard Deviation 
def ind_stdev(dados, window=1): 
    stdev_column = dados["Close"].rolling(window=window).std()
    dados["Stdev_{}".format(window)] = stdev_column
    return dados

# Volatility Index (VIX)
def ta_vix(dados, window=21):
    returns = dados["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100  
    dados["VIX_{}".format(window)] = vix
    return dados

# On-Balance Volume (OBV)
def ind_obv(dados, window=10):
    price_changes = dados["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (dados["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    dados["OBV_{}".format(window)] = obv_smoothed
    return dados

# Chaikin Money Flow (CMF)
def ind_chaikin_money_flow(dados, window=10):
    mf_multiplier = ((dados["Close"] - dados["Close"].shift(1)) + (dados["Close"] - dados["Close"].shift(1)).abs()) / 2
    mf_volume = mf_multiplier * dados["Volume"]
    adl = mf_volume.cumsum()
    cmf = adl.rolling(window=window).mean() / dados["Volume"].rolling(window=window).mean()
    dados["CMF_{}".format(window)] = cmf
    return dados

# Volume Price Trend (VPT)
def ind_volume_price_trend(dados, window=10):
    price_change = dados["Close"].pct_change()
    vpt = (price_change * dados["Volume"].shift(window)).cumsum()
    dados["VPT_{}".format(window)] = vpt
    return dados

# Accumulation/Distribution Line
def ind_accumulation_distribution_line(dados, window=10):
    money_flow_multiplier = ((dados["Close"] - dados["Close"].shift(1)) - (dados["Close"].shift(1) - dados["Close"])) / (dados["Close"].shift(1) - dados["Close"])
    money_flow_volume = money_flow_multiplier * dados["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    dados["A/D Line_{}".format(window)] = ad_line_smoothed
    return dados

# Ease of Movement (EOM)
def ind_ease_of_movement(dados, window=14):
    midpoint_move = ((dados["High"] + dados["Low"]) / 2).diff(1)
    box_ratio = dados["Volume"] / 1000000 / (dados["High"] - dados["Low"])
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    dados["EOM_{}".format(window)] = eom_smoothed
    return dados
    

