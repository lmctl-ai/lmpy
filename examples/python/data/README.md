# Data for the Python-recipe pairs

The CSV files here are exported from the datasets bundled with statsmodels 0.14.6 and scikit-learn 1.7.2. The scripts
`export-statsmodels-datasets.py` and `export-sklearn.py` regenerate them (run them with `native/.deps/oracle-venv`). No
file was downloaded from elsewhere. The repository is public, so only datasets whose terms allow redistribution are
included.

| File | Source | Licence / terms | Attribution |
|---|---|---|---|
| longley.csv | statsmodels `longley` | public domain | J. W. Longley (1967), "An Appraisal of Least Squares Programs for the Electronic Computer from the Point of View of the User", JASA 62 |
| sunspots.csv | statsmodels `sunspots` | public domain | Yearly sunspot numbers, 1700-2008 (SIDC / NOAA) |
| grunfeld.csv | statsmodels `grunfeld` | public domain | Y. Grunfeld (1958), investment data for 11 US firms |
| randhie.csv | statsmodels `randhie` | public domain | RAND Health Insurance Experiment (Deb and Trivedi, 2002) |
| anes96.csv | statsmodels `anes96` | public domain | American National Election Studies, 1996 |
| engel.csv | statsmodels `engel` | public domain | E. Engel (1857), Belgian household food expenditure |
| iris.csv, iris-split-*.csv | scikit-learn `load_iris` | CC BY 4.0 (UCI) | R. A. Fisher (1936); UCI Machine Learning Repository, Iris, doi:10.24432/C56C76 |
| wine.csv, wine-split-*.csv | scikit-learn `load_wine` | CC BY 4.0 (UCI) | S. Aeberhard and M. Forina (1991); UCI Machine Learning Repository, Wine, doi:10.24432/C5PC7J |
| breast_cancer.csv, breast_cancer-split-*.csv | scikit-learn `load_breast_cancer` | CC BY 4.0 (UCI) | W. Wolberg, O. Mangasarian, N. Street, W. Street (1993); UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), doi:10.24432/C5DW2B |
| diabetes.csv | scikit-learn `load_diabetes` | distributed with scikit-learn (BSD-3-Clause) | B. Efron, T. Hastie, I. Johnstone, R. Tibshirani (2004), "Least Angle Regression", Annals of Statistics 32 |
| blobs-*.csv | scikit-learn `make_blobs(..., random_state=170)` | generated | scikit-learn gallery, "Demonstration of k-means assumptions" |

Excluded on purpose: statsmodels' `star98` and `scotland`, whose terms say "used with express permission from the
original author, who retains all rights".

The `*-split-*` files record `train_test_split`'s row order (part 0 = training rows in order, part 1 = test rows), so
the Python and lmpy sides fit and score exactly the same rows.
