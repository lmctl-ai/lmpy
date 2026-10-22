# Source: scikit-learn 1.7 user guide 1.9.1 "Gaussian Naive Bayes" (https://scikit-learn.org/stable/modules/naive_bayes.html):
#   iris, train_test_split(test_size=0.5, random_state=0), GaussianNB, "Number of mislabeled points out of a total 75
#   points : 4"; and the gallery "Importance of Feature Scaling"
#   (https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) on the wine dataset:
#   train_test_split(test_size=0.30, random_state=42), KNeighborsClassifier(n_neighbors=20) on the (proline, hue) pair
#   with and without StandardScaler, and PCA(n_components=2) without/with StandardScaler followed by GaussianNB (the
#   classifier of the example's earlier releases; 1.7 uses LogisticRegressionCV, which lmpy does not have), reporting
#   held-out accuracy and log-loss. Extension: LinearDiscriminantAnalysis, GaussianNB and KNeighborsClassifier() on all
#   thirteen wine features of the same split.
# Prints the answers as one JSON object; examples/python/sklearn-nb-lda-knn.lms pins them.
import json

from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ans = {}
# user guide: GaussianNB on iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
ans["iris_gnb_test_size"] = X_test.shape[0]
ans["iris_gnb_mislabeled"] = int((y_test != y_pred).sum())
ans["iris_gnb_theta"] = gnb.theta_.tolist()
ans["iris_gnb_var"] = gnb.var_.tolist()
ans["iris_gnb_proba_test3"] = gnb.predict_proba(X_test[:3]).tolist()

# gallery: Importance of Feature Scaling (wine)
X, y = load_wine(return_X_y=True, as_frame=True)
scaler = StandardScaler().set_output(transform="pandas")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)

# k-neighbours on (proline, hue), fitted on all rows as the example's decision-boundary plots
X_plot = X[["proline", "hue"]]
X_plot_scaled = scaler.fit_transform(X_plot)
clf = KNeighborsClassifier(n_neighbors=20)
for name, Xp in [("knn20_unscaled", X_plot), ("knn20_scaled", X_plot_scaled)]:
    clf.fit(Xp, y)
    ans[f"{name}_train_accuracy"] = clf.score(Xp, y)

# PCA then GaussianNB, unscaled and scaled
unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB()).fit(X_train, y_train)
scaled_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB()).fit(X_train, y_train)
for name, model in [("unscaled_pca_gnb", unscaled_clf), ("scaled_pca_gnb", scaled_clf)]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    ans[f"{name}_accuracy"] = accuracy_score(y_test, y_pred)
    ans[f"{name}_log_loss"] = log_loss(y_test, y_proba)
    ans[f"{name}_explained_variance_ratio"] = model[-2].explained_variance_ratio_.tolist()

# extension: LDA, GaussianNB and kNN on all features of the same split
for name, model in [("lda", LinearDiscriminantAnalysis()), ("gnb", GaussianNB()), ("knn", KNeighborsClassifier()),
                    ("scaled_knn", make_pipeline(StandardScaler(), KNeighborsClassifier()))]:
    model.fit(X_train, y_train)
    ans[f"wine_{name}_accuracy"] = model.score(X_test, y_test)
    ans[f"wine_{name}_predict"] = model.predict(X_test).tolist()
    ans[f"wine_{name}_proba_test3"] = model.predict_proba(X_test[:3]).tolist()

print(json.dumps(ans))
