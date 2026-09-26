from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

FEATURE_LABELS = {
    "age": "Age",
    "MonthlyIncome": "Monthly income",
    "RevolvingUtilizationOfUnsecuredLines" : "Credit utilisation",
    "NumberOfTime30-59DaysPastDueNotWorse": "Late 30-59 days",
    "NumberOfTime60-89DaysPastDueNotWorse": "Late 60-89 days",
    "NumberOfTimes90DaysLate": "Late 90+ days",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfDependents": "Dependents",
    "DebtRatio": "Debt ratio",
    "debt_amount": "Debt amount",
    "sentinel_code": "Sentinel code",
    "dependents_missing": "Dependents missing",
    "income_missing": "Income missing"
}

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob) 

    return cm,rc,pr,auc

def plot_coefficients(model, X_train):

    coefficients = model.coef_[0]
    feature_names = X_train.columns

    coef_series = pd.Series(coefficients, index=feature_names).sort_values().rename(FEATURE_LABELS)

    colors = ["#D55E00" if v > 0 else "#009E73" for v in coef_series]

    coef_series.plot(kind='barh', figsize=(10, 6), color=colors)
    plt.title("Logistic regression coefficients")
    plt.xlabel("Coefficient")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig("reports/figures/model_coefficients.png", dpi=150 , bbox_inches="tight")
    plt.show()


def plot_roc_curve(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test,y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color="#1F77B4")
    plt.plot([0,1],[0,1], linestyle="--", color="grey")
    plt.title(f"ROC Curve (AUC = {auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True,alpha=0.3)
    plt.savefig("reports/figures/roc_curve.png", dpi=150 , bbox_inches="tight")
    plt.show()