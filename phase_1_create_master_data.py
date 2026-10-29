
import yfinance as yf
import pandas as pd
from phase_5_sentiment_analysis import get_recent_news, get_sentiment_average_for_news


ticker_list = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]
start_date = "2018-01-01"
end_date = "2026-10-25"


def save_master_data_to_csv(data, test=False):
    if test:
        data.to_csv(f"master_data_test.csv", index_label="Date")
    else:
        data.to_csv(f"master_data.csv", index_label="Date")

def get_ohlc_data(ticker):
    start = start_date
    end = end_date
    
    data = yf.download(ticker, start=start, end=end)
    data.columns = data.columns.get_level_values(0)
    data['Ticker'] = ticker
    data['daily_returns'] = daily_returns(data)
    data['open_direction'] = open_direction(data)
    data['rsi_14'] = relative_strength_index(data, 14)
    data['adx_14'] = average_directional_index(data, 14)
    data['macd'] = moving_average_convergence_divergence(data)
    data['obv'] = on_balance_volume(data)
    data['sp500_daily_returns'] = sp500_daily_returns(start, end)
    data['vix_close'] = vix_close(start, end)
    data['Quarter'] = data.index.to_series().dt.quarter
    data['Month'] = data.index.to_series().dt.month
    data['Year'] = data.index.to_series().dt.year
    data['sentiment'] = 0  # initialize sentiment column with 0, will be updated later with actual sentiment values

    # drop the warm-up rows where indicators are not yet computable
    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'daily_returns', 'open_direction', 'rsi_14', 'adx_14', 'macd', 'obv', 'sp500_daily_returns', 'vix_close', 'Quarter', 'Month', 'Year', 'sentiment']]
    return data.dropna()

def daily_returns(ohlc_data):
    data = ohlc_data['Close'].pct_change() # change from previous value
    # multiply by 100 to get percentage
    data = data * 100
    data.iloc[0] = 0 # adding zero for the first day since there is no previous day to compare
    return data

def open_direction(ohlc_data):
    # Open DirecSon (defined at t)=1 if Open (t) > Close (t-1); else 0
    data = ohlc_data['Open'] - ohlc_data['Close'].shift(1) # change from previous value
    data = (data > 0).astype(int) # 1 if Open > previous Close, else 0
    data.iloc[0] = 0 # adding zero for the first day since there is no previous day to compare
    return data

def relative_strength_index(ohlc_data, period=14):
    delta = ohlc_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def average_directional_index(ohlc_data, period=14):
    high = ohlc_data['High']
    low = ohlc_data['Low']
    close = ohlc_data['Close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    smoothing = {'alpha': 1 / period, 'adjust': False, 'min_periods': period}
    atr = tr.ewm(**smoothing).mean()
    plus_di = 100 * plus_dm.ewm(**smoothing).mean() / atr
    minus_di = 100 * minus_dm.ewm(**smoothing).mean() / atr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0)
    return dx.ewm(**smoothing).mean()

def moving_average_convergence_divergence(ohlc_data, span_short=12, span_long=26):
    ema_short = ohlc_data['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = ohlc_data['Close'].ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    return macd

def on_balance_volume(ohlc_data):
    close_change = ohlc_data['Close'].diff()
    direction = close_change.gt(0).astype(int) - close_change.lt(0).astype(int)
    return (direction * ohlc_data['Volume']).cumsum()

def sp500_daily_returns(start, end):
    sp500_data = yf.download("^GSPC", start=start, end=end)
    sp500_data.columns = sp500_data.columns.get_level_values(0)
    return daily_returns(sp500_data)

def vix_close(start, end):
    vix_data = yf.download("^VIX", start=start, end=end)
    vix_data.columns = vix_data.columns.get_level_values(0)
    return vix_data['Close'].rename('vix_close')

# -----------------------------------------

def load_master_data():
    master_data = pd.DataFrame()

    for ticker in ticker_list:
        # Merge all the above files and create a master data
        data = get_ohlc_data(ticker)
        master_data = pd.concat([master_data, data], axis=0)
        
        news = get_recent_news(ticker)
        sentiment = get_sentiment_average_for_news(news)
        # update master data with sentiment values for the correspondind dates
        for s in sentiment.itertuples():
            matching_rows = (master_data['Ticker'] == ticker) & (master_data.index.date == s.date)
            master_data.loc[matching_rows, 'sentiment'] = s.average_sentiment
        
    save_master_data_to_csv(master_data)
