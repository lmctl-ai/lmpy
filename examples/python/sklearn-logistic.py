# Source: scikit-learn 1.7 LogisticRegression reference example
#   (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): iris, multinomial,
#   predict / predict_proba of the first two rows and the training accuracy; and the "Getting Started" pipeline
#   (https://scikit-learn.org/stable/getting_started.html): make_pipeline(StandardScaler(), LogisticRegression()) with
#   train_test_split(random_state=0) and accuracy_score on the held-out rows, run on iris (as in the guide) and on the
#   breast_cancer dataset. Extension: each model also solved to convergence (lbfgs tol 1e-12, max_iter 10000) as the
#   *_conv_* keys, since the published defaults stop lbfgs early (iris: at max_iter=100 with a ConvergenceWarning).
# Prints the answers as one JSON object; examples/python/sklearn-logistic.lms pins them.
import json
import warnings

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The reference example stops at lbfgs's max_iter=100 on unscaled iris (a ConvergenceWarning, as in the published output);
# keep stdout to the JSON answers.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
ans = {}
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
ans["iris_predict"] = clf.predict(X[:2, :]).tolist()
ans["iris_proba"] = clf.predict_proba(X[:2, :]).tolist()
ans["iris_score"] = clf.score(X, y)
ans["iris_coef"] = clf.coef_.tolist()
ans["iris_intercept"] = clf.intercept_.tolist()
ans["iris_proba_rows_50_100"] = clf.predict_proba(X[[50, 100], :]).tolist()
# extension: the same model solved to convergence (the optimum the example's lbfgs run is approaching)
clf = LogisticRegression(max_iter=10000, tol=1e-12).fit(X, y)
ans["iris_conv_coef"] = clf.coef_.tolist()
ans["iris_conv_intercept"] = clf.intercept_.tolist()
ans["iris_conv_proba"] = clf.predict_proba(X[[0, 1, 50, 100], :]).tolist()
ans["iris_conv_score"] = clf.score(X, y)

for name, loader in [("iris_pipe", load_iris), ("cancer", load_breast_cancer)]:
    X, y = loader(return_X_y=True)
    # create a pipeline object
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipe.fit(X_train, y_train)
    ans[f"{name}_accuracy"] = accuracy_score(pipe.predict(X_test), y_test)
    lr = pipe[-1]
    ans[f"{name}_coef"] = lr.coef_.tolist()
    ans[f"{name}_intercept"] = lr.intercept_.tolist()
    ans[f"{name}_proba_test5"] = pipe.predict_proba(X_test[:5]).tolist()
    ans[f"{name}_scaler_mean"] = pipe[0].mean_.tolist()
    ans[f"{name}_scaler_scale"] = pipe[0].scale_.tolist()
    # extension: the same pipeline solved to convergence
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, tol=1e-12)).fit(X_train, y_train)
    ans[f"{name}_conv_coef"] = pipe[-1].coef_.tolist()
    ans[f"{name}_conv_intercept"] = pipe[-1].intercept_.tolist()
    ans[f"{name}_conv_proba_test5"] = pipe.predict_proba(X_test[:5]).tolist()

print(json.dumps(ans))
