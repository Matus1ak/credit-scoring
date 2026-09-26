from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    rc = recall_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob) 

    return cm,rc,pr,auc