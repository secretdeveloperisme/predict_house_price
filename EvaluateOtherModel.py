import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from LinearRegression import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression as LinearRegressionLibrary
housing_price_data = pd.read_csv("data/HousingPrices-Amsterdam-August-2021.csv")

# clean data
housing_price_data.dropna(inplace=True)
X = housing_price_data.iloc[:, [2, 4, 5, 6, 7]].copy()
X["Zip"] = X["Zip"].str.extract("(\d+)").astype(float)
X = np.array(X)
# print(X.head().to_string())
y = housing_price_data["Price"]
y = np.array(y)
housing_price_data.drop(["Price"], axis=1, inplace=True)


mse_file = open("evaluate/evaluate_mse.csv", mode="w", newline="")
fieldnames = ['Loop', 'LinearRegression', 'LinearRegressionLib', 'Decision Tree', 'Ridge', 'Lasso', 'BayesianRidge']
mse_csv = csv.DictWriter(mse_file, fieldnames=fieldnames)
mse_csv.writeheader()
score_file = open("evaluate/evaluate_r_squared.csv", mode="w", newline="")
score_csv = csv.DictWriter(score_file, fieldnames=fieldnames)
score_csv.writeheader()
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3 / 10.0, random_state=i+20)

    linear_model = LinearRegression()
    linear_model_lib = LinearRegressionLibrary()
    ridge_model = Ridge()
    lasso_model = Lasso()
    decision_tree_model = DecisionTreeRegressor(max_depth=200)
    br_model = BayesianRidge()

    linear_model_lib.fit(X_train, y_train)
    linear_model.fit_stochastic(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    decision_tree_model.fit(X_train, y_train)
    br_model.fit(X_train, y_train)

    linear_lib_mse = mean_squared_error(y_true=y_test, y_pred=linear_model_lib.predict(X_test))
    linear_mse = mean_squared_error(y_true=y_test, y_pred=linear_model.predict(X_test))
    ridge_mse = mean_squared_error(y_true=y_test, y_pred=ridge_model.predict(X_test))
    lasso_mse = mean_squared_error(y_true=y_test, y_pred=lasso_model.predict(X_test))
    decision_tree_mse = mean_squared_error(y_true=y_test, y_pred=decision_tree_model.predict(X_test))
    br_mse = mean_squared_error(y_true=y_test, y_pred=br_model.predict(X_test))

    linear_lib_score = linear_model_lib.score(X_test,y_test)
    linear_score = linear_model.r2_score(y_test, linear_model.predict(X_test))
    ridge_score = ridge_model.score(X_test, y_test)
    lasso_score = lasso_model.score(X_test, y_test)
    decision_tree_score = decision_tree_model.score(X_test, y_test)
    br_score = br_model.score(X_test, y_test)
    print(decision_tree_score)
    mse_csv.writerow({
        "Loop": i,
        "LinearRegression": linear_mse,
        "LinearRegressionLib": linear_lib_mse,
        "Decision Tree": decision_tree_mse,
        "Ridge": ridge_mse,
        "Lasso": lasso_mse,
        "BayesianRidge": br_mse
    })
    score_csv.writerow({
        "Loop": i,
        "LinearRegression": round(linear_score*100, 3),
        "LinearRegressionLib": round(linear_lib_score*100, 3),
        "Decision Tree": round(decision_tree_score * 100, 3),
        "Ridge": round(ridge_score*100, 3),
        "Lasso": round(lasso_score*100, 3),
        "BayesianRidge": round(br_score*100, 3)
    })
mse_file.close()
score_file.close()
