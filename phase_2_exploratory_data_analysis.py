import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import f_oneway

# Figure factory: plain Figure is thread-safe (used by the API); the script mode swaps it for pyplot
make_figure = Figure


def load_eda_data(path='master_data.csv'):
    data = pd.read_csv(path)
    # New collumn for Year-Month combination
    data['YearMonth'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    return data


def plot_time_series(data, column, title, ylabel):
    # Line chart of a column over the years, one line per ticker
    fig = make_figure(figsize=(12, 6))
    ax = fig.subplots()
    sns.lineplot(data=data, x='YearMonth', y=column, hue='Ticker', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Year / Month')
    ax.set_xticks(range(0, len(data['YearMonth'].unique()), 3))
    ax.set_xticklabels(data['YearMonth'].unique()[::3], rotation=45)
    ax.set_ylabel(ylabel)
    ax.legend(title='Technology Stock')
    fig.tight_layout()
    return fig


def compute_ranking(data):
    # Ranking of technology stocks based on returns and risk (volatility)
    returns = data.groupby('Ticker')['Close'].pct_change().groupby(data['Ticker']).mean()
    volatility = data.groupby('Ticker')['Close'].pct_change().groupby(data['Ticker']).std()
    ranking = pd.DataFrame({'Returns': returns, 'Volatility': volatility})
    ranking['Sharpe_Ratio'] = ranking['Returns'] / ranking['Volatility']
    return ranking.sort_values(by='Sharpe_Ratio', ascending=False)


def compute_beta(data):
    # How these stocks are sensitive to benchmark index (S&P 500)
    # Both return series are already daily returns (%), so no further pct_change is needed
    return data.groupby('Ticker').apply(
        lambda g: g['daily_returns'].cov(g['sp500_daily_returns']) / g['sp500_daily_returns'].var()
    )


def plot_bar(series, title, ylabel):
    fig = make_figure(figsize=(10, 6))
    ax = fig.subplots()
    sns.barplot(x=series.index, y=series.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Ticker')
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


# Bivariate Analysis using ANOVA: bivariate relationships between features and Stock Open Direction
target = 'open_direction'
excluded_columns = ['Date', 'YearMonth', 'Ticker', target]

def bivariate_analysis(df, alpha=0.05, plot=False):
    # ANOVA F-test of each numeric feature across the Open Direction groups (0 vs 1), per stock
    features = [c for c in df.select_dtypes(include='number').columns if c not in excluded_columns]
    results = []
    for ticker, group in df.groupby('Ticker'):
        for feature in features:
            down = group.loc[group[target] == 0, feature].dropna()
            up = group.loc[group[target] == 1, feature].dropna()
            if len(down) < 2 or len(up) < 2:
                continue
            f_stat, p_value = f_oneway(down, up)
            results.append({
                'Ticker': ticker,
                'Feature': feature,
                'Mean_Direction_0': down.mean(),
                'Mean_Direction_1': up.mean(),
                'F_statistic': f_stat,
                'p_value': p_value,
                'Significant': p_value < alpha,
            })
    results = pd.DataFrame(results).sort_values(['Ticker', 'p_value']).reset_index(drop=True)

    if plot:
        plot_bivariate_heatmap(results)
    return results

def plot_bivariate_heatmap(results):
    # heatmap: brighter = stronger relationship with Open Direction
    pivot = results.pivot(index='Feature', columns='Ticker', values='p_value')
    fig = make_figure(figsize=(10, 8))
    ax = fig.subplots()
    sns.heatmap(-np.log10(pivot.clip(lower=1e-300)), annot=True, fmt='.1f', cmap='viridis', ax=ax)
    ax.set_title('Bivariate ANOVA vs Open Direction')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    make_figure = plt.figure
    data = load_eda_data()
    print(data.head())
    print(data.info())
    plot_time_series(data, 'Close', 'Closing Prices Over the Years (2018-2026)', 'Closing Price')
    plot_time_series(data, 'Volume', 'Trading Volume Over the Years (2018-2026)', 'Volume')
    ranking = compute_ranking(data)
    print(ranking)
    plot_bar(ranking['Sharpe_Ratio'], 'Technology Stocks Ranking Based on Sharpe Ratio', 'Sharpe Ratio')
    beta = compute_beta(data)
    print(beta)
    plot_bar(beta, 'Technology Stocks Sensitivity to S&P 500 (Beta)', 'Beta')
    bivariate_results = bivariate_analysis(data, plot=True)
    print(bivariate_results.to_string(index=False))
    plt.show()
