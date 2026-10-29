import base64
import html
import io
import threading
from phase_1_create_master_data import load_master_data
from phase_3and4_predictive_modeling_ml_methods import load_predictions
from phase_5_sentiment_analysis import get_recent_news, get_sentiment
import phase_2_exploratory_data_analysis as eda
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()
loaded = False
load_error = None
predictions_next_day = {}


def load_all():
    global loaded, load_error, predictions_next_day
    try:
        load_master_data()
        predictions_next_day = load_predictions()
        loaded = True
        print("Models loaded successfully.")
    except Exception as exc:
        load_error = str(exc)
        raise


# Models are trained in the background so the UI can show a loader meanwhile
threading.Thread(target=load_all, daemon=True).start()


def require_loaded():
    if not loaded:
        raise HTTPException(status_code=503, detail="Models are still loading")


@app.get("/status")
def status():
    return {"loaded": loaded, "error": load_error}


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
def to_python(value):
    # Model outputs are numpy scalars, which FastAPI cannot always serialise
    return value.item() if hasattr(value, "item") else value


@app.get("/predict")
def predictions():
    require_loaded()
    # The prediction is calculated once (since the data does not change) when the script is run, and returned here.
    return {ticker: to_python(value) for ticker, value in predictions_next_day.items()}


# -------------------------------------------------
# Ticker details endpoint: next-day prediction, overall sentiment and latest articles
# -------------------------------------------------
@app.get("/ticker-info/{ticker}")
def ticker_info(ticker: str):
    require_loaded()
    ticker = ticker.upper()
    if ticker not in predictions_next_day:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    news = get_recent_news(ticker)
    news["sentiment"] = news["title"].apply(lambda x: get_sentiment(x) if x else 0)
    overall = float(news["sentiment"].mean()) if len(news) else 0.0
    label = "Positive" if overall > 0 else "Negative" if overall < 0 else "Neutral"

    prediction = to_python(predictions_next_day[ticker])
    articles = [
        {
            "title": row.title,
            "url": row.url,
            "publisher": row.publisher,
            "published": row.published,
            "sentiment": float(row.sentiment),
        }
        for row in news.itertuples()
    ]
    return {
        "ticker": ticker,
        "prediction": prediction,
        "prediction_label": "Up" if prediction == 1 else "Down",
        "overall_sentiment": overall,
        "overall_sentiment_label": label,
        "articles": articles,
    }


# -------------------------------------------------
# HTML frontend embedded directly in endpoint
# -------------------------------------------------
@app.get("/predict-ui", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Next-Day Open Direction Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; max-width: 900px; }
            select { padding: 8px 12px; font-size: 16px; margin-left: 8px; }
            .card { border: 1px solid #ccc; border-radius: 6px; padding: 16px; margin-top: 20px; }
            .up { color: #0a7a2f; font-weight: bold; }
            .down { color: #b00020; font-weight: bold; }
            #prediction { font-size: 1.8em; }
            li { margin-bottom: 8px; }
            .loader { border: 4px solid #eee; border-top-color: #3b82f6; border-radius: 50%;
                      width: 28px; height: 28px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; }
            @keyframes spin { to { transform: rotate(360deg); } }
            #loading { margin: 20px 0; }
            .meta { color: #666; font-size: 13px; }
        </style>
    </head>
    <body>
        <h2>Next-Day Open Direction Prediction</h2>

        <div id="loading"><span class="loader"></span> <span id="loadingText">Loading models, please wait...</span></div>

        <div id="selector" style="display:none">
            <label for="ticker">Ticker:</label>
            <select id="ticker" onchange="loadTicker()">
                <option value="" disabled selected>Select a ticker</option>
            </select>
        </div>

        <div id="result" class="card" style="display:none">
            <p>Prediction for next day: <span id="prediction"></span></p>
            <p>Overall sentiment: <span id="sentiment"></span></p>
            <h3>Latest articles</h3>
            <ul id="articles"></ul>
        </div>

        <script>
            function showReady() {
                fetch('/predict')
                    .then(response => response.json())
                    .then(data => {
                        const select = document.getElementById('ticker');
                        Object.keys(data).forEach(t => {
                            const option = document.createElement('option');
                            option.value = t;
                            option.textContent = t;
                            select.appendChild(option);
                        });
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('selector').style.display = 'block';
                    });
            }

            function waitUntilLoaded() {
                fetch('/status')
                    .then(response => response.json())
                    .then(status => {
                        if (status.error) {
                            document.querySelector('#loading .loader').style.display = 'none';
                            document.getElementById('loadingText').textContent = 'Loading failed: ' + status.error;
                        } else if (status.loaded) {
                            showReady();
                        } else {
                            setTimeout(waitUntilLoaded, 2000);
                        }
                    })
                    .catch(() => setTimeout(waitUntilLoaded, 2000));
            }
            waitUntilLoaded();

            function loadTicker() {
                const ticker = document.getElementById('ticker').value;
                fetch('/ticker-info/' + encodeURIComponent(ticker))
                    .then(response => {
                        if (!response.ok) throw new Error('HTTP ' + response.status);
                        return response.json();
                    })
                    .then(data => {
                        const pred = document.getElementById('prediction');
                        pred.textContent = data.prediction_label;
                        pred.className = data.prediction_label === 'Up' ? 'up' : 'down';
                        document.getElementById('sentiment').textContent =
                            data.overall_sentiment_label + ' (' + data.overall_sentiment.toFixed(3) + ')';

                        const list = document.getElementById('articles');
                        list.innerHTML = '';
                        if (data.articles.length === 0) {
                            list.textContent = 'No recent articles found.';
                        }
                        data.articles.forEach(a => {
                            const li = document.createElement('li');
                            const link = document.createElement('a');
                            link.textContent = a.title || '(untitled)';
                            if (a.url && /^https?:/i.test(a.url)) link.href = a.url;
                            link.target = '_blank';
                            link.rel = 'noopener noreferrer';
                            const meta = document.createElement('div');
                            meta.className = 'meta';
                            meta.textContent = [a.publisher, a.published, 'sentiment ' + a.sentiment.toFixed(2)]
                                .filter(Boolean).join(' | ');
                            li.appendChild(link);
                            li.appendChild(meta);
                            list.appendChild(li);
                        });
                        document.getElementById('result').style.display = 'block';
                    })
                    .catch(error => {
                        alert('Error fetching data');
                        console.error(error);
                    });
            }
        </script>

    </body>
    </html>
    """

    return html_content


# -------------------------------------------------
# Phase 2: exploratory data analysis (graphs and tables)
# -------------------------------------------------
def figure_to_img(fig, alt):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{encoded}">'


@app.get("/eda-content", response_class=HTMLResponse)
def eda_content():
    require_loaded()
    data = eda.load_eda_data()
    ranking = eda.compute_ranking(data)
    beta = eda.compute_beta(data).rename("Beta").to_frame()
    bivariate = eda.bivariate_analysis(data)

    sections = [
        ("Closing prices over the years",
         figure_to_img(eda.plot_time_series(data, "Close", "Closing Prices Over the Years (2018-2026)", "Closing Price"), "Closing prices")),
        ("Trading volume over the years",
         figure_to_img(eda.plot_time_series(data, "Volume", "Trading Volume Over the Years (2018-2026)", "Volume"), "Trading volume")),
        ("Ranking by return and risk",
         ranking.to_html(float_format="{:.4f}".format)
         + figure_to_img(eda.plot_bar(ranking["Sharpe_Ratio"], "Technology Stocks Ranking Based on Sharpe Ratio", "Sharpe Ratio"), "Sharpe ratio ranking")),
        ("Sensitivity to the S&P 500 (beta)",
         figure_to_img(eda.plot_bar(beta["Beta"], "Technology Stocks Sensitivity to S&P 500 (Beta)", "Beta"), "Beta")),
        ("Bivariate analysis (ANOVA) vs Open Direction",
         figure_to_img(eda.plot_bivariate_heatmap(bivariate), "ANOVA heatmap")
         + bivariate.to_html(index=False, float_format="{:.4g}".format)),
    ]
    return "".join(f"<h2>{title}</h2>{content}" for title, content in sections)


@app.get("/eda-ui", response_class=HTMLResponse)
def eda_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 2 - Exploratory Data Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; max-width: 1000px; }
            img { max-width: 100%; display: block; margin: 16px 0; }
            table { border-collapse: collapse; margin: 12px 0; }
            th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: right; }
            th { background-color: #f4f4f4; }
            .loader { border: 4px solid #eee; border-top-color: #3b82f6; border-radius: 50%;
                      width: 28px; height: 28px; animation: spin 1s linear infinite; display: inline-block; vertical-align: middle; }
            @keyframes spin { to { transform: rotate(360deg); } }
            #loading { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>Phase 2 - Exploratory Data Analysis</h1>
        <div id="loading"><span class="loader"></span> <span id="loadingText">Loading models, please wait...</span></div>
        <div id="content"></div>

        <script>
            function showError(message) {
                document.querySelector('#loading .loader').style.display = 'none';
                document.getElementById('loadingText').textContent = message;
            }

            function loadContent() {
                document.getElementById('loadingText').textContent = 'Generating charts...';
                fetch('/eda-content')
                    .then(response => {
                        if (!response.ok) throw new Error('HTTP ' + response.status);
                        return response.text();
                    })
                    .then(content => {
                        document.getElementById('content').innerHTML = content;
                        document.getElementById('loading').style.display = 'none';
                    })
                    .catch(error => showError('Error fetching data: ' + error.message));
            }

            function waitUntilLoaded() {
                fetch('/status')
                    .then(response => response.json())
                    .then(status => {
                        if (status.error) {
                            showError('Loading failed: ' + status.error);
                        } else if (status.loaded) {
                            loadContent();
                        } else {
                            setTimeout(waitUntilLoaded, 2000);
                        }
                    })
                    .catch(() => setTimeout(waitUntilLoaded, 2000));
            }
            waitUntilLoaded();
        </script>
    </body>
    </html>
    """
