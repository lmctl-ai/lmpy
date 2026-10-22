# Exports the scikit-learn data the sklearn-*.py recipes use to CSV, so the lmpy versions (sklearn-*.lms) read the same
# numbers. Run with pycli's oracle venv (scikit-learn 1.7.2):
#   native/.deps/oracle-venv/bin/python examples/python/data/export-sklearn.py
# Provenance: iris, wine, breast_cancer and diabetes are the datasets bundled with scikit-learn (sklearn.datasets.load_*,
# distributed under scikit-learn's BSD-3-Clause licence; originally from the UCI Machine Learning Repository, and Efron et
# al. 2004 "Least Angle Regression" for diabetes, here in sklearn's default scaled form). Column names are sklearn's
# feature names with non-alphanumerics replaced by "_"; `target` is the class code / regression target.
# blobs-1500-rs170.csv is make_blobs(n_samples=1500, random_state=170) as in the gallery's "Demonstration of k-means
# assumptions", and blobs-varied-1500-rs170.csv its make_blobs(..., cluster_std=[1.0, 2.5, 0.5], random_state=170)
# (make_blobs draws from numpy's legacy RandomState, so it is exported rather than regenerated).
# The *-split-*.csv files record train_test_split's row order for the recipes' own calls (part 0 = the X_train rows in
# order, part 1 = the X_test rows in order), so both sides fit and score the same rows in the same order.
import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, make_blobs
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))


def clean(name):
    out = "".join(c if c.isalnum() else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def dataset(loader, name):
    d = loader()
    f = pd.DataFrame(d.data, columns=[clean(c) for c in d.feature_names])
    f["target"] = d.target
    f.to_csv(os.path.join(HERE, name + ".csv"), index=False)
    return len(d.target)


def split(name, n, **kw):
    tr, te = train_test_split(np.arange(n), **kw)
    pd.DataFrame({"part": [0] * len(tr) + [1] * len(te), "row": np.concatenate([tr, te])}).to_csv(
        os.path.join(HERE, name + ".csv"), index=False)


n = dataset(load_iris, "iris")
split("iris-split-rs0-t25", n, random_state=0)                   # getting-started pipeline (default test_size 0.25)
split("iris-split-rs0-t50", n, test_size=0.5, random_state=0)    # user guide GaussianNB
n = dataset(load_wine, "wine")
split("wine-split-rs42-t30", n, test_size=0.30, random_state=42)  # gallery "Importance of Feature Scaling"
n = dataset(load_breast_cancer, "breast_cancer")
split("breast_cancer-split-rs0-t25", n, random_state=0)
dataset(load_diabetes, "diabetes")
Xb, yb = make_blobs(n_samples=1500, random_state=170)
pd.DataFrame({"x0": Xb[:, 0], "x1": Xb[:, 1], "y": yb}).to_csv(os.path.join(HERE, "blobs-1500-rs170.csv"), index=False)
Xv, yv = make_blobs(n_samples=1500, cluster_std=[1.0, 2.5, 0.5], random_state=170)
pd.DataFrame({"x0": Xv[:, 0], "x1": Xv[:, 1], "y": yv}).to_csv(os.path.join(HERE, "blobs-varied-1500-rs170.csv"), index=False)
