from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import warnings
import  mplfinance as mpf
import plotly.graph_objects as go

def settings() :
    """ some settings for better output looking"""
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.width', None)
settings()


end_date = datetime.datetime.now()
end_date = end_date.strftime('%Y-%m-%d')
start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 10)
ticker = '  ' # Write the name of the stock whose indicator values you want to calculate in the blank.
df = yf.download(tickers=ticker, start=start_date, end=end_date)

################    RSI    ################
def calculate_rsi(data, period=14):
    """TradingView ile uyumlu RSI hesaplama"""

    def rma(series, period):
        """Running Moving Average (RMA) hesaplama"""
        alpha = 1 / period
        rma_values = [series[0]]
        for price in series[1:]:
            rma_values.append(alpha * price + (1 - alpha) * rma_values[-1])
        return np.array(rma_values)

    
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)

    rs = avg_gain / avg_loss
    rsi = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    return rsi

df['RSI'] = calculate_rsi(df, period=14)

################    ATR    ################
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr
df['atr'] = calculate_atr(df)
################    MOMENTUM    ################
def calculate_momentum(df, period=14):
    momentum = df['Close'].diff(period)
    return momentum
df['momentum'] = calculate_momentum(df, period=14)
################    MACD    ################
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal
df['MACD'], df['MACD_Signal'] = calculate_macd(df)
################    CMO    ################
def calculate_cmo(df, period=14):
    delta = df['Close'].diff()    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)    
    sum_gain = gain.rolling(window=period, min_periods=1).sum()
    sum_loss = loss.rolling(window=period, min_periods=1).sum()  
    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)
    return cmo
df['CMO'] = calculate_cmo(df, period=14)
################    CCI    ################
def calculate_cci(df, period=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    return cci
df['CCI'] = calculate_cci(df, period=20)
################    MFI    ################
def calculate_mfi(df, period=14):

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi
df['MFI'] = calculate_mfi(df, period=14)
################    EMA'S    ################
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
################    BOLLINGER BANDS    ################

def calculate_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    bb_low = sma - (num_std * rolling_std)
    bb_high = sma + (num_std * rolling_std)
    bb_mid = sma
    return bb_low, bb_mid, bb_high

df['bb_low'], df['bb_mid'], df['bb_high'] = calculate_bollinger_bands(df['Close'], window=20, num_std=2)
####################################################################

print(df.tail(20))
