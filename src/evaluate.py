from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd

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

    coef_series = pd.Series(coefficients, index=feature_names).sort_values()

    coef_series.plot(kind='barh', figsize=(8, 6))
    plt.title("Logistic regression coefficients")
    plt.xlabel("Coefficient")
    plt.savefig("reports/figures/model_coefficients.png", dpi=150 , bbox_inches="tight")
    plt.show()


def plot_roc_curve(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--", color="grey")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("reports/figures/roc_curve.png", dpi=150 , bbox_inches="tight")
    plt.show()