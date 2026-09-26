from src.data import load_and_clean
from src.features import split_data, impute_missing_values, scale_features
from src.models import train_model
from src.evaluate import evaluate_model, plot_coefficients, plot_roc_curve

df = load_and_clean("data/cs-training.csv")
X_train, X_test, y_train, y_test = split_data(df)
X_train, X_test = impute_missing_values(X_train, X_test)
X_train, X_test = scale_features(X_train, X_test)
model = train_model(X_train, y_train)

cm, rc, pr, auc = evaluate_model(model, X_test, y_test)

print("Confusion Matrix:\n", cm, "\n")

print("AUC:", auc)
print("Recall:", rc)
print("Precision:", pr)

plot_coefficients(model, X_train)
plot_roc_curve(model, X_test, y_test)