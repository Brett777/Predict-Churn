from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    mean_squared_error,
    median_absolute_error,
    f1_score,
    r2_score,
    recall_score,
    roc_curve,
)
import numpy as np

# metric methods


def mse(y_test, y_pred):
    return {"key": "mse", "value": mean_squared_error(y_test, y_pred)}


def r2(y_test, y_pred):
    return {"key": "r2", "value": r2_score(y_test, y_pred)}


def f1(y_test, y_pred):
    return {"key": "f1", "value": f1_score(y_test, y_pred)}


def precision(y_test, y_pred):
    return {"key": "precision", "value": precision_score(y_test, y_pred)}


def accuracy(y_test, y_pred):
    return {"key": "accuracy", "value": accuracy_score(y_test, y_pred)}


def recall(y_test, y_pred):
    return {"key": "recall", "value": recall_score(y_test, y_pred)}


# diagnostics methods


def residuals(y_test, y_pred):
    return {
        "type": "residuals",
        "residuals": (
            (np.array(y_pred) - np.array(y_test)) / np.array(y_test)
        ).tolist(),
        "observations": y_test,
    }


def roc_curve_dict(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    return {
        "type": "roc_curve",
        "truePositiveRates": tpr.tolist(),
        "falsePositiveRates": fpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def confusion_matrix_dict(y_test, y_pred):
    return {
        "type": "confusion_matrix",
        "matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


class Evaluator(object):
    # default methods. 'metrics' return float type, 'diagnostics' return numpy array
    evaluation_methods = {
        "regression": {"metrics": [mse, r2], "diagnostics": [residuals]},
        "binary_classification": {
            "metrics": [f1, accuracy, precision, recall],
            "diagnostics": [confusion_matrix_dict, roc_curve_dict],
        },
    }

    def __init__(self, problem_class):
        # validate problem_class
        if problem_class not in self.evaluation_methods:
            raise NotImplementedError(
                "Problem class " + problem_class + " is not recognised."
            )
        else:
            self.problem_class = problem_class
            self.metrics = self.evaluation_methods[self.problem_class]["metrics"]
            self.diagnostics = self.evaluation_methods[self.problem_class][
                "diagnostics"
            ]
