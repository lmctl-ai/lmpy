# Source: scikit-learn 1.7 user guide 3.1 "Cross-validation: evaluating estimator performance"
#   (https://scikit-learn.org/stable/modules/cross_validation.html): iris, cross_val_score(clf, X, y, cv=5) (integer cv on
#   a classifier = StratifiedKFold without shuffling) and the StratifiedKFold / KFold splitters; and the cross_val_score
#   reference example (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html):
#   Lasso() on the first 150 diabetes rows with cv=3 (-> [0.3315057 0.08022103 0.03531816]).
#   The user guide's svm.SVC(kernel='linear', C=1) is not in lmpy; the recipe scores the classifiers lmpy has instead:
#   LogisticRegression, KNeighborsClassifier, LinearSVC, LinearDiscriminantAnalysis and GaussianNB (sklearn defaults,
#   except max_iter=1000 for LogisticRegression on unscaled iris so lbfgs converges).
# Prints the answers as one JSON object; examples/python/sklearn-cross-val.lms pins them.
import json

from sklearn import datasets, linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

ans = {}
X, y = datasets.load_iris(return_X_y=True)
models = [("logistic", LogisticRegression(max_iter=1000)), ("knn", KNeighborsClassifier()), ("linear_svc", LinearSVC()),
          ("lda", LinearDiscriminantAnalysis()), ("gnb", GaussianNB())]
for name, clf in models:
    scores = cross_val_score(clf, X, y, cv=5)
    ans[f"cv5_{name}"] = scores.tolist()
    ans[f"cv5_{name}_mean"] = scores.mean()
    ans[f"cv5_{name}_std"] = scores.std()

# the splitters themselves: test-fold index of every row
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold = [0] * len(y)
for k, (_, test) in enumerate(skf.split(X, y)):
    for i in test:
        fold[i] = k
ans["skf_shuffle_test_fold"] = fold
for name, clf in models:
    ans[f"skf_shuffle_{name}"] = cross_val_score(clf, X, y, cv=skf).tolist()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ans["kfold_shuffle_knn"] = cross_val_score(KNeighborsClassifier(), X, y, cv=kf).tolist()

# cross_val_score reference example
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
ans["diabetes_lasso_cv3"] = cross_val_score(lasso, X, y, cv=3).tolist()

print(json.dumps(ans))
