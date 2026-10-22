# Source: scikit-learn 1.7 user guide 1.1 "Linear Models" (https://scikit-learn.org/stable/modules/linear_model.html):
#   the LinearRegression (1.1.1), Ridge(alpha=.5) (1.1.2) and Lasso(alpha=0.1) (1.1.3) snippets, verbatim; and the gallery
#   example "Linear Regression Example" (https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html):
#   diabetes, one feature (bmi), the last 20 rows held out (train_test_split(test_size=20, shuffle=False)), mse and R^2.
#   Extension: the same split with all ten diabetes features for LinearRegression and the user guide's Ridge/Lasso alphas.
# Prints the answers as one JSON object; examples/python/sklearn-linear-diabetes.lms pins them.
import json

from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ans = {}
# 1.1.1 Ordinary Least Squares
reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
ans["ug_ols_coef"] = reg.coef_.tolist()
# 1.1.2 Ridge regression
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
ans["ug_ridge_coef"] = reg.coef_.tolist()
ans["ug_ridge_intercept"] = reg.intercept_
# 1.1.3 Lasso
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
ans["ug_lasso_predict"] = reg.predict([[1, 1]]).tolist()

# plot_ols: Linear Regression Example
X, y = load_diabetes(return_X_y=True)
X1 = X[:, [2]]  # Use only one feature
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=20, shuffle=False)
regressor = LinearRegression().fit(X_train, y_train)
y_pred = regressor.predict(X_test)
ans["ols_bmi_coef"] = regressor.coef_.tolist()
ans["ols_bmi_intercept"] = regressor.intercept_
ans["ols_bmi_mse"] = mean_squared_error(y_test, y_pred)
ans["ols_bmi_r2"] = r2_score(y_test, y_pred)

# extension: all ten features, same split, the user guide's estimators
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, shuffle=False)
for name, model in [("ols", LinearRegression()), ("ridge", linear_model.Ridge(alpha=.5)),
                    ("lasso", linear_model.Lasso(alpha=0.1))]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ans[f"{name}_coef"] = model.coef_.tolist()
    ans[f"{name}_intercept"] = float(model.intercept_)
    ans[f"{name}_mse"] = mean_squared_error(y_test, y_pred)
    ans[f"{name}_r2"] = r2_score(y_test, y_pred)
    ans[f"{name}_train_r2"] = model.score(X_train, y_train)

print(json.dumps(ans))
