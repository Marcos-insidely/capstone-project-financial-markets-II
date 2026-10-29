
============================
INSTALLTION AND INSTRUCTIONS
============================

The Python code for this project was deployed to Render.com, using Github as the repository for its source code storage.
GitHub link: https://github.com/Marcos-insidely/capstone-project-financial-markets-II

Render.com public web access
Project’s base URL: https://capstone-project-financial-markets-ii.onrender.com
All the available endpoint paths are listed below.


=========
ENDPOINTS
=========

Main endpoints
/predict-ui: Provides a UI interface for selecting a ticker to show next day opening, overall market sentiment and a list of the latest news about the stock.
/eda-ui: Provides data analysis for the group of tech stocks (* It take ages to load on Render.com, works better locally), with graphs and tables.

Support endpoints
/predict: Provides the final prediction of stock openings for the next day (JSON format).
/ticker-info/{ticker}: Provides the same info as the /predict-ui endpoint, but in JSON format.
/status: Enables the activity indicator shown in other pages.


===========================
RUNNING THE PROJECT LOCALLY
===========================

The script load_api.py is the project main entry point, it loads and executes the other scripts, depending on the endpoint requested; they do not need to be executed individually or in advance.
It uses FastAPI to provide a web interface to show information about the selected tech stocks: "AAPL", "AMZN", "GOOG", "MSFT", "NVDA".

Before runnig load_api.py, it is necessary to install all of its dependencies using the following command:
`pip install -r requirements.txt`

Once all dependencies are installed, the script can be invoked using the uvicorn tool:
`uvicorn load_api:app --host 0.0.0.0 --port 10000`

This will start the FastAPI API local endpoint at http://127.0.0.1:8000, that can be viewed in any web browser.