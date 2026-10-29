import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timezone
import pandas as pd

TICKER_LIST = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]

def parse_ticker_news(item):
    # Ticker.news items nest the data under "content"
    content = item.get("content", item)
    url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url")
    return {
        "published": content.get("pubDate"),
        "title": content.get("title"),
        "summary": content.get("summary"),
        "publisher": (content.get("provider") or {}).get("displayName"),
        "url": url,
    }


def parse_search_news(item):
    published = item.get("providerPublishTime")
    if published is not None:
        published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
    return {
        "published": published,
        "title": item.get("title"),
        "summary": item.get("summary"), # None,
        "publisher": item.get("publisher"),
        "url": item.get("link"),
    }


def get_recent_news(ticker, count=20):
    """Return recent Yahoo Finance articles about a ticker as a DataFrame."""
    items = yf.Ticker(ticker).get_news(count=count, tab="news")
    rows = [parse_ticker_news(i) for i in items]

    # Ticker.news is sometimes empty, so fall back to the search endpoint
    if not rows:
        found = yf.Search(ticker, news_count=count).news
        rows = [parse_search_news(i) for i in found if ticker in i.get("relatedTickers", [])]

    df = pd.DataFrame(rows, columns=["published", "title", "summary", "publisher", "url"])
    df.insert(0, "ticker", ticker)
    return df


def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_sentiment_average_for_news(df):
    if "sentiment" not in df.columns:
        df["sentiment"] = df["title"].apply(lambda x: get_sentiment(x) if x else 0)
    # return a dataframe of average sentiment values for each diffefrent day in the dataframe. if sentiment < 0 it becomes -1, if sentiment > 0 it becomes 1, otherwise 0
    df["date"] = pd.to_datetime(df["published"]).dt.date
    daily_sentiment = df.groupby("date")["sentiment"].mean()
    daily_sentiment = daily_sentiment.apply(lambda x: -1 if x < 0 else (1 if x > 0 else 0))
    return daily_sentiment.reset_index().rename(columns={"sentiment": "average_sentiment"})


for ticker in TICKER_LIST:
    news = get_recent_news(ticker)
    # print(f"{ticker}: {len(news)} articles")
    # print(get_sentiment_average_for_news(news))



