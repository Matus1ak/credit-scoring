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


    # The counters are imputed with 0, the median and mode of these columns.
    # This is only safe because sentinel_code carries the risk signal: without
    # that flag these 269 borrowers would look like clients who never missed
    # a payment, while over half of them default.
    df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].fillna(0)
    df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].fillna(0)
    df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].fillna(0)


    # Missing dependents are not random: those rows default at 4.56% against
    # 6.74%, so the absence is flagged before it is imputed with 0.
    df["dependents_missing"] = (df["NumberOfDependents"].isna()).astype(int)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)


    # Above 5 dependents the group sizes fall below a hundred records and the
    # default rate becomes erratic, so the upper values are collapsed into one.
    df["NumberOfDependents"] = df["NumberOfDependents"].clip(upper=5)


    # Where income is missing, DebtRatio holds a debt amount rather than a
    # ratio, so the amount is moved to its own column and the ratio cleared.
    df["income_missing"] = (df["MonthlyIncome"].isna()).astype(int)
    df["debt_amount"] = df["DebtRatio"].where(df["MonthlyIncome"].isna())
    df["DebtRatio"] = df["DebtRatio"].mask(df["MonthlyIncome"].isna())


    # Borrowers who reported an income have no debt amount recorded here, so 0
    # means the quantity does not apply rather than that it is unknown. The
    # income_missing flag tells the two cases apart.
    df["debt_amount"] = df["debt_amount"].fillna(0)


    # 241 records exceed 10, with a default rate close to the baseline, so the
    # values are capped rather than flagged or removed.
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10)

    

    return df