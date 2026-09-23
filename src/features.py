import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df):

    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def impute_missing_values(X_train, X_test):

    income_median = X_train["MonthlyIncome"].median()
    X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(income_median)
    X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(income_median)

    # DebtRatio is unknown wherever income was not reported, so it is imputed
    # with the training median, computed after the split to avoid leakage.
    debt_ratio_median = X_train["DebtRatio"].median()
    X_train["DebtRatio"] = X_train["DebtRatio"].fillna(debt_ratio_median)
    X_test["DebtRatio"] = X_test["DebtRatio"].fillna(debt_ratio_median)

    return X_train, X_test