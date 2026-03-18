from sklearn.metrics import accuracy_score, r2_score


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    # Detect problem type
    if y_test.nunique() < 20:
        score = accuracy_score(y_test, predictions)
    else:
        score = r2_score(y_test, predictions)

    return score