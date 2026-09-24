from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train,y_train)

    return model