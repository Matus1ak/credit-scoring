import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_SCALE = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "debt_amount"
]


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

def scale_features(X_train, X_test):

    # Statistics are learned from the training set only and applied to both,
    # so no information from the test set reaches the model. Binary flags are
    # left unscaled to keep their coefficients directly interpretable.
    scaler = StandardScaler()
    X_train[COLUMNS_TO_SCALE] = scaler.fit_transform(X_train[COLUMNS_TO_SCALE])
    X_test[COLUMNS_TO_SCALE] = scaler.transform(X_test[COLUMNS_TO_SCALE])

    return X_train, X_test