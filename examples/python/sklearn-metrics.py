# Source: scikit-learn 1.7 user guide 3.4 "Metrics and scoring" (https://scikit-learn.org/stable/modules/model_evaluation.html)
#   and the reference examples of sklearn.metrics (https://scikit-learn.org/stable/api/sklearn.metrics.html), verbatim inputs:
#   accuracy_score, confusion_matrix (numeric and the animal labels), classification_report (output_dict=True),
#   precision_recall_fscore_support (cat/dog/pig, macro/micro/weighted), balanced_accuracy_score, roc_auc_score,
#   log_loss (spam/ham), r2_score, mean_squared_error, mean_absolute_error.
# Prints the answers as one JSON object; examples/python/sklearn-metrics.lms pins them.
import json

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss,
                             mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score,
                             roc_auc_score)

ans = {}
# accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
ans["accuracy"] = accuracy_score(y_true, y_pred)
ans["accuracy_count"] = float(accuracy_score(y_true, y_pred, normalize=False))

# confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
ans["confusion"] = confusion_matrix(y_true, y_pred).tolist()
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
ans["confusion_animals"] = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"]).tolist()
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel().tolist()
ans["binary_tn_fp_fn_tp"] = [tn, fp, fn, tp]

# classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
rep = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
for k, name in enumerate(target_names):
    ans[f"report_class{k}"] = [rep[name]["precision"], rep[name]["recall"], rep[name]["f1-score"], rep[name]["support"]]
ans["report_accuracy"] = rep["accuracy"]
ans["report_macro"] = [rep["macro avg"][m] for m in ("precision", "recall", "f1-score")]
ans["report_weighted"] = [rep["weighted avg"][m] for m in ("precision", "recall", "f1-score")]

# precision_recall_fscore_support
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
for avg in ("macro", "micro", "weighted"):
    ans[f"prf_{avg}"] = list(precision_recall_fscore_support(y_true, y_pred, average=avg)[:3])

# balanced_accuracy_score
ans["balanced_accuracy"] = balanced_accuracy_score([0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1])

# roc_auc_score (roc_curve example data)
ans["roc_auc"] = roc_auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]))

# log_loss
ans["log_loss"] = log_loss(["spam", "ham", "ham", "spam"], [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])

# regression metrics
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
ans["r2"] = r2_score(y_true, y_pred)
ans["mse"] = mean_squared_error(y_true, y_pred)
ans["mae"] = mean_absolute_error(y_true, y_pred)

print(json.dumps(ans))
