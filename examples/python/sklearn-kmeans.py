# Source: scikit-learn 1.7 KMeans reference example
#   (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html): six 2-D points,
#   KMeans(n_clusters=2, random_state=0, n_init="auto") -> labels_, predict([[0, 0], [12, 3]]), cluster_centers_; and the
#   gallery "Demonstration of k-means assumptions"
#   (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html): make_blobs(1500, random_state=170)
#   clustered with the wrong k, after an anisotropic transformation, with unequal variances and with unevenly sized blobs,
#   then the uneven blobs again with n_init=10. Extension: iris with KMeans(n_clusters=3, random_state=0, n_init=10) and
#   with init="random", n_init=1.
# For each fit the recipe reports labels, inertia_, cluster_centers_ and n_iter_.
# Prints the answers as one JSON object; examples/python/sklearn-kmeans.lms pins them.
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs

ans = {}


def record(name, km):
    ans[f"{name}_labels"] = km.labels_.tolist()
    ans[f"{name}_inertia"] = km.inertia_
    ans[f"{name}_centers"] = km.cluster_centers_.tolist()
    ans[f"{name}_n_iter"] = int(km.n_iter_)


X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
record("doc", kmeans)
ans["doc_predict"] = kmeans.predict([[0, 0], [12, 3]]).tolist()

n_samples = 1500
random_state = 170
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

X, y = make_blobs(n_samples=n_samples, random_state=random_state)
X_aniso = np.dot(X, transformation)  # Anisotropic blobs
X_varied, y_varied = make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)  # Unequal variance
X_filtered = np.vstack(
    (X[y == 0][:500], X[y == 1][:100], X[y == 2][:10])
)  # Unevenly sized blobs
y_filtered = [0] * 500 + [1] * 100 + [2] * 10

common_params = {
    "n_init": "auto",
    "random_state": random_state,
}
record("wrong_k", KMeans(n_clusters=2, **common_params).fit(X))
record("aniso", KMeans(n_clusters=3, **common_params).fit(X_aniso))
record("varied", KMeans(n_clusters=3, **common_params).fit(X_varied))
record("filtered", KMeans(n_clusters=3, **common_params).fit(X_filtered))
record("filtered_ninit10", KMeans(n_clusters=3, n_init=10, random_state=random_state).fit(X_filtered))

X = load_iris().data
record("iris", KMeans(n_clusters=3, random_state=0, n_init=10).fit(X))
record("iris_random_init", KMeans(n_clusters=3, init="random", n_init=1, random_state=0).fit(X))

print(json.dumps(ans))
