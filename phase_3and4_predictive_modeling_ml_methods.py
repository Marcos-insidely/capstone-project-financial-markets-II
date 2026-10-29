
from unittest import result

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import statsmodels.api as sm

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


TICKER_LIST = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]
TARGET = 'open_direction'


def build_features(csv_path, ticker):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df[df['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

    # Convert categorical columns to category dtype
    df['Ticker'] = df['Ticker'].astype('category')

    result = pd.DataFrame() #({'Date': df['Date']})
    
    result['Quarter'] = df['Quarter'].astype('category')
    result['Month'] = df['Month'].astype('category')
    result['Year'] = df['Year'].astype('category')

    # print(df.head())
    # print(df.info())

    result['open_direction'] = df['open_direction']
    # returns of the previous 3 days
    result['ret_t1'] = df['daily_returns'].shift(1)
    result['ret_t2'] = df['daily_returns'].shift(2)
    result['ret_t3'] = df['daily_returns'].shift(3)
    # technical indicators and market variables as of the previous day
    result['obv_t1'] = df['obv'].shift(1)
    result['Open_t1'] = df['Open'].shift(1)
    result['High_t1'] = df['High'].shift(1)
    result['Low_t1'] = df['Low'].shift(1)
    result['Close_t1'] = df['Close'].shift(1)
    result['Volume_t1'] = df['Volume'].shift(1)
    result['rsi_t1'] = df['rsi_14'].shift(1)
    result['adx_t1'] = df['adx_14'].shift(1)
    result['macd_t1'] = df['macd'].shift(1)
    result['sp500_ret_t1'] = df['sp500_daily_returns'].shift(1)
    result['vix_close_t1'] = df['vix_close'].shift(1)
    result['sentiment_t1'] = df['sentiment'].shift(1)
    # extra feature: previous day High/Low ratio
    result['high_low_ratio_t1'] = (df['High'] / df['Low']).shift(1)

    # drop X first 3 rows due to lagging
    result = result.iloc[3:]

    return result


def roc_auc(model, X_train_A, y_train, X_test_A, y_test):
    # ROC Curve for Train and Test Data
    y_train_pred_prob = model.predict_proba(X_train_A)[:,1]
    y_test_pred_prob = model.predict_proba(X_test_A)[:,1]
    
    train_auc = roc_auc_score(y_train, y_train_pred_prob)
    test_auc = roc_auc_score(y_test, y_test_pred_prob)
    # print("Train AUC Score:", round(train_auc,4))
    # print("Test AUC Score:", round(test_auc,4))
    
    # Find Optimum Threshold for classification by maximizing sensitivity + specificity
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred_prob)
    test_fpr, test_tpr, _ = roc_curve(y_test, y_test_pred_prob)

    # Build ROC dataframe
    roc = pd.DataFrame({
            'fpr'        : train_fpr,
            'tpr'        : train_tpr,
            '1-fpr'      : 1 - train_fpr,
            'tf'         : train_tpr - train_fpr,
            'thresholds' : train_thresholds
    })
    
    # Find optimal threshold
    optimal_idx = np.argmax(train_tpr - train_fpr)
    optimal_threshold = roc.iloc[optimal_idx]['thresholds']

    # print("Optimal Threshold:", round(optimal_threshold,4))
    # print(f"ROC points — Train: {len(train_fpr)}, Test: {len(test_fpr)}")
    
    # Apply Threshold — Train Confusion Matrix & Classification Report
    # Predict based on threshold
    y_train_pred_opt = (y_train_pred_prob >= optimal_threshold).astype(int)
    
    # print("\nTRAIN CONFUSION MATRIX:")
    # print(confusion_matrix(y_train, y_train_pred_opt))
    # print("\nTRAIN CLASSIFICATION REPORT:")
    # print(classification_report(y_train, y_train_pred_opt))
    
    # Apply Threshold — Test Confusion Matrix & Classification Report
    y_test_pred_opt = (y_test_pred_prob >= optimal_threshold).astype(int)
    
    # print("\nTEST CONFUSION MATRIX:")
    # print(confusion_matrix(y_test, y_test_pred_opt))
    # print("\nTEST CLASSIFICATION REPORT:")
    # print(classification_report(y_test, y_test_pred_opt))

    return y_train_pred_opt, y_test_pred_opt, train_auc, test_auc

    
def create_logistic_regression_model(TICKER):
    df = build_features('master_data.csv', TICKER)

    # Split the data into training and testing sets (80/20)
    train_size = int(len(df) * 0.8)
    df_test = df.iloc[train_size:].copy()
    df = df.iloc[:train_size].copy()

    # Scale variables if necessary (e.g., using StandardScaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Features constant in the training set (e.g. sentiment) carry no information and make the logit singular
    feature_cols = [c for c in df.columns if c != 'open_direction' and df[c].nunique() > 1]
    dropped_constant = [c for c in df.columns if c != 'open_direction' and c not in feature_cols]
    if dropped_constant:
        # print(f"Dropped constant training features for {TICKER}: {dropped_constant}")
        pass
    # Scaler is fitted on the training set only and then applied to the test set
    scaler.fit(df[feature_cols])
    df_test_scaled = pd.DataFrame(scaler.transform(df_test[feature_cols]), columns=feature_cols, index=df_test.index)
    df_test_scaled['open_direction'] = df_test['open_direction']
    df_scaled = pd.DataFrame(scaler.transform(df[feature_cols]), columns=feature_cols, index=df.index)
    df_scaled['open_direction'] = df['open_direction']
    df = df_scaled
    df_test = df_test_scaled

    X = df.drop(['open_direction'], axis=1) # df[significant_vars]
    X_test = df_test.drop(['open_direction'], axis=1) # df_test[significant_vars]
    X = sm.add_constant(X)
    X_test = sm.add_constant(X_test)

    # Check Multicollinearity
    # Check correlation among features
    corr_matrix = X.corr()
    # print(corr_matrix)

    # Identify highly correlated features
    high_corr_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.8]
    # print("Highly correlated feature pairs:", high_corr_pairs)

    # Group features that are (directly or indirectly) highly correlated, so each group is combined into one component.
    groups = []
    for col1, col2 in high_corr_pairs:
        merged = [g for g in groups if col1 in g or col2 in g]
        new_group = {col1, col2}.union(*merged)
        groups = [g for g in groups if g not in merged] + [new_group]

    # Combine each group of highly correlated features into a single feature, using PCA
    for group in groups:
        cols = sorted(group)
        # PCA is fitted on the training set only and applied to the test set
        pca = PCA(n_components=1).fit(X[cols])
        X[f"{'_'.join(cols)}_pca"] = pca.transform(X[cols])
        X_test[f"{'_'.join(cols)}_pca"] = pca.transform(X_test[cols])

    # Drop combined columns after PCA transformation
    columns_to_drop = [col for group in groups for col in group]
    X.drop(columns=columns_to_drop, inplace=True)
    X_test.drop(columns=columns_to_drop, inplace=True)

    # Remove predictors with excessive VIF after PCA, fitting only on training data.
    vif_threshold = 5.0
    removed_for_vif = []
    while len(X.columns) > 1:
        vif_data = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns,
            name='VIF'
        )
        predictor_vif = vif_data.drop(index='const', errors='ignore')
        high_vif = predictor_vif[predictor_vif > vif_threshold]
        if high_vif.empty:
            break

        feature_to_remove = high_vif.idxmax()
        removed_for_vif.append(feature_to_remove)
        X.drop(columns=feature_to_remove, inplace=True)
        X_test.drop(columns=feature_to_remove, inplace=True)

    vif_data = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
        name='VIF'
    )
    # print(f"Predictors removed for VIF > {vif_threshold:g}: {removed_for_vif}")
    # print(f"Training VIF after multicollinearity handling for {TICKER}:")
    # print(vif_data.to_string())

    # ---------------------------------------------

    # Test the remaining predictors after PCA, then refit using significant ones.
    X['open_direction'] = df['open_direction']
    candidate_vars = [col for col in X.columns if col not in ('const', 'open_direction')]
    if candidate_vars:
        candidate_formula = 'open_direction ~ ' + ' + '.join(candidate_vars)
        candidate_model = smf.logit(candidate_formula, data=X).fit()
        significant_vars = candidate_model.pvalues.drop('Intercept', errors='ignore')
        significant_vars = significant_vars[significant_vars < 0.05].index.tolist()
    else:
        significant_vars = []
    # print(f"Significant predictors for {TICKER} (p < 0.05): {significant_vars}")

    if significant_vars:
        final_formula = 'open_direction ~ ' + ' + '.join(significant_vars)
    else:
        # print(f"No predictors met p < 0.05 for {TICKER}; fitting an intercept-only model.")
        final_formula = 'open_direction ~ 1'
    model = smf.logit(final_formula, data=X).fit()
    if significant_vars:
        final_vif = pd.Series(
            [variance_inflation_factor(X[['const', *significant_vars]].values, i)
             for i in range(len(significant_vars) + 1)],
            index=['const', *significant_vars],
            name='VIF'
        )
        # print(f"Final-model training VIF for {TICKER}:")
        # print(final_vif.to_string())

    # ---------------------------------------------
    
    # Predict Probabilities
    y_train = df['open_direction']
    y_test = df_test['open_direction']
    p_train = model.predict(X)
    p_test = model.predict(X_test)

    # Obtain ROC curve and AUC for train data
    # Obtain threshold to balance sensiSvity and specificity
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train, p_train)
    test_fpr, test_tpr, _ = roc_curve(y_test, p_test)
    threshold = train_thresholds[np.argmax(train_tpr - train_fpr)]
    # print(f"ROC points — Train: {len(train_fpr)}, Test: {len(test_fpr)}")
    # print(f"Train size: {len(y_train)}, test size: {len(y_test)}, threshold: {threshold:.3f}")

    for name, y, p in [("TRAIN", y_train, p_train), ("TEST", y_test, p_test)]:
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        # print(f"{name}: AUC={roc_auc_score(y, p):.3f} accuracy={accuracy_score(y, pred):.3f} "
            #   f"sensitivity={tp / (tp + fn):.3f} specificity={tn / (tn + fp):.3f} (share of up-opens={y.mean():.3f})")

    # ---------------------------------------------

    return model, y_test, p_test, roc_auc_score(y_test, p_test)


def create_ml_base_processing(TICKER):
    masterdata = build_features('master_data.csv', TICKER)

    # build_features returns rows sorted by date; keep the latest 20% for testing.
    train_size = int(len(masterdata) * 0.8)
    train_data = masterdata.iloc[:train_size].copy()
    test_data = masterdata.iloc[train_size:].copy()

    bucket_vars = ['ret_t1', 'ret_t2', 'ret_t3', 'obv_t1', 'Open_t1', 'High_t1', 'Low_t1', 'Close_t1', 'Volume_t1', 'rsi_t1', 'adx_t1', 'macd_t1', 'sp500_ret_t1', 'vix_close_t1', 'sentiment_t1', 'high_low_ratio_t1']

    X_train = train_data[bucket_vars]
    X_test = test_data[bucket_vars]
    y_train = train_data['open_direction']
    y_test = test_data['open_direction']

    # Display shapes
    # print("Training Data Shape:", X_train.shape)
    # print("Testing Data Shape:", X_test.shape)
    # print("\nTrain Response Distribution:\n", y_train.value_counts(normalize=True)*100)
    # print("\nTest Response Distribution:\n", y_test.value_counts(normalize=True)*100)

    return X_train, y_train, X_test, y_test


def create_naive_bayes_model(TICKER):
    X_train_A, y_train, X_test_A, y_test = create_ml_base_processing(TICKER)

    gnb = GaussianNB()
    gnb.fit(X_train_A, y_train)

    _, y_test_pred, _, test_auc = roc_auc(gnb, X_train_A, y_train, X_test_A, y_test)
    return gnb, y_test, y_test_pred, test_auc


def create_decision_tree_model(TICKER):
    X_train_A, y_train, X_test_A, y_test = create_ml_base_processing(TICKER)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_A, y_train)

    _, y_test_pred, _, test_auc = roc_auc(dt, X_train_A, y_train, X_test_A, y_test)
    return dt, y_test, y_test_pred, test_auc


def create_random_forest_model(TICKER):
    X_train_A, y_train, X_test_A, y_test = create_ml_base_processing(TICKER)

    rf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train_A, y_train)

    _, y_test_pred, _, test_auc = roc_auc(rf, X_train_A, y_train, X_test_A, y_test)
    return rf, y_test, y_test_pred, test_auc


def create_boosting_model(TICKER):
    X_train_A, y_train, X_test_A, y_test = create_ml_base_processing(TICKER)

    boost = GradientBoostingClassifier(n_estimators=300, random_state=42)
    boost.fit(X_train_A, y_train)

    _, y_test_pred, _, test_auc = roc_auc(boost, X_train_A, y_train, X_test_A, y_test)
    return boost, y_test, y_test_pred, test_auc


def create_svm_model(TICKER):
    X_train_A, y_train, X_test_A, y_test = create_ml_base_processing(TICKER)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_A)
    X_test_scaled = scaler.transform(X_test_A)

    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    _, y_test_pred, _, test_auc = roc_auc(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)
    return svm_model, y_test, y_test_pred, test_auc


# ==============================================================
# Models and Predictions
# ==============================================================

def load_predictions():
    model_functions = [
        ("Logistic Regression", create_logistic_regression_model),
        ("Naive Bayes", create_naive_bayes_model),
        ("Decision Tree", create_decision_tree_model),
        ("Random Forest", create_random_forest_model),
        ("Boosting", create_boosting_model),
        ("SVM", create_svm_model),
    ]
    model_results = []

    for TICKER in TICKER_LIST:
        for model_name, model_function in model_functions:
            model, y_test, predictions, test_auc = model_function(TICKER)
            model_results.append({
                'Ticker': TICKER,
                'Model': model_name,
                'Test_AUC': test_auc,
            })
            result = pd.DataFrame({'test_data': y_test, 'prediction': predictions})
            # print("-------------------------------------------------------------------------------------------")
            # print(f"{model_name} Model for {TICKER} generates the following dataFrame:\n{result.head()}")

    model_comparison = pd.DataFrame(model_results).sort_values(
        ['Ticker', 'Test_AUC'],
        ascending=[True, False]
    ).reset_index(drop=True)
    # print("\nTEST AUC COMPARISON (ranked within each ticker):")
    # print(model_comparison.to_string(index=False, formatters={'Test_AUC': '{:.4f}'.format}))

    best_models = model_comparison.groupby('Ticker', sort=False).head(1)
    # print("\nSELECTED MODEL PER TICKER (highest held-out test AUC):")
    # print(best_models.to_string(index=False, formatters={'Test_AUC': '{:.4f}'.format}))


    # save the best models for each ticker in memory for later use
    best_models_dict = {}
    for TICKER in TICKER_LIST:
        best_model_name = best_models.loc[best_models['Ticker'] == TICKER, 'Model'].values[0]
        for model_name, model_function in model_functions:
            if model_name == best_model_name:
                model, _, _, _ = model_function(TICKER)
                best_models_dict[TICKER] = model
                break

    master_data = pd.read_csv("master_data.csv", index_col=0)
    # use the best models to predict open_direction for the next day
    predictions_next_day = {}
    for TICKER in TICKER_LIST:
        model = best_models_dict[TICKER]
        ticker_data = master_data[master_data['Ticker'] == TICKER].copy()
        if not ticker_data.empty:
            X_next_day = ticker_data.iloc[[-1]].drop(columns=['Ticker', 'open_direction'], errors='ignore')
            scaler = StandardScaler()
            X_next_day_scaled = scaler.fit_transform(X_next_day)
            predictions_next_day[TICKER] = model.predict(X_next_day_scaled)[0]

    print("Predictions for the next day:")
    print(predictions_next_day)

    return predictions_next_day