import pandas as pd

def load_and_clean(file_path):

    df = pd.read_csv(file_path,index_col=0)

    # A single record has age 0, which is not a possible value; drop it.
    df = df[df["age"] > 0]


    # Values of 96 and 98 are sentinel codes, not counts. The flag is created
    # before they are removed, since afterwards the rows are indistinguishable
    df["sentinel_code"] = (df["NumberOfTimes90DaysLate"]>90).astype(int)


    # The true number of delinquencies is unknown for these rows, so the codes
    # are replaced with missing values rather than treated as observed counts.
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].mask(df["NumberOfTimes90DaysLate"] > 90)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].mask(df["NumberOfTime60-89DaysPastDueNotWorse"] > 90)
    df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].mask(df["NumberOfTime30-59DaysPastDueNotWorse"] > 90)


    # Missing dependents are not random: those rows default at 4.56% against
    # 6.74%, so the absence is flagged before it is imputed with 0.
    df["dependents_missing"] = (df["NumberOfDependents"].isna()).astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)


    # Where income is missing, DebtRatio holds a debt amount rather than a
    # ratio, so the amount is moved to its own column and the ratio cleared.
    df["income_missing"] = (df["MonthlyIncome"].isna()).astype(int)
    df["debt_amount"] = df["DebtRatio"].where(df["MonthlyIncome"].isna())
    df["DebtRatio"] = df["DebtRatio"].mask(df["MonthlyIncome"].isna())

    return df