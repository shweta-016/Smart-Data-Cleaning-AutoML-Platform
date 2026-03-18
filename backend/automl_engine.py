from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score


def run_automl(X_train, X_test, y_train, y_test):

    scores = {}
    best_model = None
    best_name = ""
    best_score = -999

    # Detect problem type
    if y_train.nunique() < 20:
        problem_type = "classification"
    else:
        problem_type = "regression"


    # ---------------- CLASSIFICATION ---------------- #

    if problem_type == "classification":

        models = {

            "Logistic Regression": (
                LogisticRegression(max_iter=500),
                {"C":[0.01,0.1,1,10]}
            ),

            "Decision Tree": (
                DecisionTreeClassifier(),
                {"max_depth":[3,5,10]}
            ),

            "Random Forest": (
                RandomForestClassifier(),
                {"n_estimators":[50,100]}
            ),

            "SVM": (
                SVC(),
                {"C":[0.1,1,10]}
            ),

            "XGBoost": (
                XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                {"n_estimators":[50,100]}
            )
        }

        for name,(model,params) in models.items():

            grid = GridSearchCV(
                model,
                params,
                cv=3,
                scoring="accuracy"
            )

            grid.fit(X_train,y_train)

            best_estimator = grid.best_estimator_

            predictions = best_estimator.predict(X_test)

            score = accuracy_score(y_test,predictions)

            scores[name] = round(score,4)

            if score > best_score:
                best_score = score
                best_model = best_estimator
                best_name = name


    # ---------------- REGRESSION ---------------- #

    else:

        models = {

            "Linear Regression":(
                LinearRegression(),
                {}
            ),

            "Decision Tree Regressor":(
                DecisionTreeRegressor(),
                {"max_depth":[3,5,10]}
            ),

            "Random Forest Regressor":(
                RandomForestRegressor(),
                {"n_estimators":[50,100]}
            ),

            "SVR":(
                SVR(),
                {"C":[0.1,1,10]}
            ),

            "XGBoost Regressor":(
                XGBRegressor(),
                {"n_estimators":[50,100]}
            )
        }

        for name,(model,params) in models.items():

            grid = GridSearchCV(
                model,
                params,
                cv=3,
                scoring="r2"
            )

            grid.fit(X_train,y_train)

            best_estimator = grid.best_estimator_

            predictions = best_estimator.predict(X_test)

            score = r2_score(y_test,predictions)

            scores[name] = round(score,4)

            if score > best_score:
                best_score = score
                best_model = best_estimator
                best_name = name


    return best_model,best_name,scores