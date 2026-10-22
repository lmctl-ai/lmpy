# Source: scikit-learn 1.7 gallery "Comparison of LDA and PCA 2D projection of Iris dataset"
#   (https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html): PCA(n_components=2) and
#   LinearDiscriminantAnalysis(n_components=2) fitted on iris, the explained variance ratios the example prints, and the
#   projected rows it plots; plus the PCA reference example
#   (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) on its six 2-D points.
# Sign convention: PCA's components_ follow svd_flip (each component's largest |loading| is positive), so they are
#   compared as printed. LDA's scalings come from LAPACK SVDs whose signs sklearn leaves unspecified; the recipe reports
#   each LDA axis multiplied by the sign of its scalings column's largest-|entry| (lmpy's convention), which is the
#   same subspace and the same plot up to a mirror.
# Prints the answers as one JSON object; examples/python/sklearn-pca-iris.lms pins them.
import json
import sys

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ans = {}
# PCA reference example
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
ans["doc_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
ans["doc_singular_values"] = pca.singular_values_.tolist()

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_), file=sys.stderr)  # the example prints this; stdout carries the JSON
ans["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
ans["pca_explained_variance"] = pca.explained_variance_.tolist()
ans["pca_singular_values"] = pca.singular_values_.tolist()
ans["pca_components"] = pca.components_.tolist()
ans["pca_mean"] = pca.mean_.tolist()
ans["pca_rows_0_50_100"] = X_r[[0, 50, 100]].tolist()
ans["pca_class_means"] = [X_r[y == k].mean(axis=0).tolist() for k in range(3)]

ans["lda_explained_variance_ratio"] = lda.explained_variance_ratio_.tolist()
sign = np.sign(lda.scalings_[np.abs(lda.scalings_).argmax(axis=0), np.arange(lda.scalings_.shape[1])])[:2]
X_r2 = X_r2 * sign
ans["lda_rows_0_50_100"] = X_r2[[0, 50, 100]].tolist()
ans["lda_class_means"] = [X_r2[y == k].mean(axis=0).tolist() for k in range(3)]
ans["lda_coef"] = lda.coef_.tolist()
ans["lda_intercept"] = lda.intercept_.tolist()
ans["lda_training_accuracy"] = lda.score(X, y)

print(json.dumps(ans))
