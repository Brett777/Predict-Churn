class SklearnObjectClass(object):
    object_class_parameters = {
        "LinearRegression": lambda model: [
            {"key": "intercept", "value": model.intercept_.item()},
            {"key": "coefficient", "value": model.coef_.tolist()},
        ],
        "DummyClassifier": lambda model: [],
        "LogisticRegression": lambda model: [
            {"key": "intercept", "value": model.intercept_.tolist()},
            {"key": "coefficient", "value": model.coef_.tolist()},
        ],
        "LogisticRegressionCV": lambda model: [
            {"key": "best estimator intercept", "value": model.intercept_.tolist()},
            {"key": "best estimator coefficient", "value": model.coef_.tolist()},
            {
                "key": "best estimator C (inverse of optimal regularization value)",
                "value": model.C_.tolist(),
            },
        ],
        "PassiveAggressiveClassifier": lambda model: [
            {"key": "intercept", "value": model.intercept_.tolist()},
            {"key": "coefficient", "value": model.coef_.tolist()},
        ],
        "RidgeClassifier": lambda model: [
            {"key": "intercept", "value": model.intercept_.tolist()},
            {"key": "coefficient", "value": model.coef_.tolist()},
        ],
        "RidgeClassifierCV": lambda model: [
            {"key": "best estimator intercept", "value": model.intercept_.tolist()},
            {"key": "best estimator coefficient", "value": model.coef_.tolist()},
        ],
        "SGDClassifier": lambda model: [
            {"key": "intercept", "value": model.intercept_.tolist()},
            {"key": "coefficient", "value": model.coef_.tolist()},
        ],
        "AdaBoostClassifier": lambda model: [
            {"key": "feature importances", "value": model.feature_importances_.tolist()}
        ],
        "BaggingClassifier": lambda model: [
            {"key": "out-of-bag score", "value": model.oob_score}
        ],
        "ExtraTreesClassifier": lambda model: [
            {
                "key": "feature importances",
                "value": model.feature_importances_.tolist(),
            },
            {
                "key": "out-of-bag score",
                "value": model.oob_score_.tolist()
                if model.get_params()["oob_score"] is True
                else "",
            },
        ],
        "GradientBoostingClassifier": lambda model: [
            {"key": "feature importances", "value": model.feature_importances_.tolist()}
        ],
        "RandomForestClassifier": lambda model: [
            {
                "key": "feature importances",
                "value": model.feature_importances_.tolist(),
            },
            {
                "key": "out-of-bag score",
                "value": model.oob_score_.tolist()
                if model.get_params()["oob_score"] is True
                else "",
            },
        ],
        "VotingClassifier": lambda model: [],
        "GaussianProcessClassifier": lambda model: [
            {
                "key": "log marginal likelihood of theta",
                "value": model.log_marginal_likelihood_value_.tolist(),
            }
        ],
        "BernoulliNB": lambda model: [
            {"key": "class log prior", "value": model.class_log_prior_.tolist()},
            {
                "key": "feature log probability",
                "value": model.feature_log_prob_.tolist(),
            },
        ],
        "GaussianNB": lambda model: [
            {"key": "class prior", "value": model.class_prior_.tolist()},
            {"key": "feature value means by class", "value": model.theta_.tolist()},
            {"key": "feature value variances by class", "value": model.sigma_.tolist()},
        ],
        "MultinomialNB": lambda model: [
            {"key": "class log prior", "value": model.class_log_prior_.tolist()},
            {
                "key": "feature log probability",
                "value": model.feature_log_prob_.tolist(),
            },
        ],
        "KNeighborsClassifier": lambda model: [],
        "RadiusNeighborsClassifier": lambda model: [],
        "NearestCentroid": lambda model: [
            {"key": "centroids", "value": model.centroids_.tolist()}
        ],
        "MLPClassifier": lambda model: [
            {
                "key": "bias vectors",
                "value": list(map(lambda x: x.tolist(), model.intercepts_)),
            },
            {"key": "weights", "value": list(map(lambda x: x.tolist(), model.coefs_))},
        ],
        "DecisionTreeClassifier": lambda model: [
            {"key": "feature importances", "value": model.feature_importances_.tolist()}
        ],
        "ExtraTreeClassifier": lambda model: [
            {"key": "feature importances", "value": model.feature_importances_.tolist()}
        ],
        # sklearn.model_selection objects:
        "GridSearchCV": lambda model: [
            {
                "key": "best estimator",
                "value": str(model.best_estimator_).replace("\n         ", ""),
            },
            {"key": "best score", "value": model.best_score_},
            # Best estimator params/hyperparams:
            {
                "key": "best estimator hyperparams",
                "value": model.best_estimator_.get_params(),
            },
            {
                "key": "best estimator coefficients",
                "value": model.best_estimator_.coef_.tolist()
                if hasattr(model.best_estimator_, "coef_")
                else "",
            },
        ],
        "RandomizedSearchCV": lambda model: [
            {
                "key": "best estimator",
                "value": str(model.best_estimator_).replace("\n         ", ""),
            },
            {"key": "best score", "value": model.best_score_},
            # Best estimator params/hyperparams:
            {
                "key": "best estimator hyperparams",
                "value": model.best_estimator_.get_params(),
            },
            {
                "key": "best estimator coefficients",
                "value": model.best_estimator_.coef_.tolist()
                if hasattr(model.best_estimator_, "coef_")
                else "",
            },
        ],
        # sklearn.pipeline object:
        "Pipeline": lambda model: [
            {
                "key": "pipeline steps",
                "value": {
                    key: str(val).replace("\n ", "")
                    for key, val in model.named_steps.items()
                },
            },
            {
                "key": "estimator coefficients",
                "value": model.coef_.tolist() if hasattr(model, "coef_") else "",
            },
        ],
    }

    def __init__(self, object_class):
        if object_class not in self.object_class_parameters:
            raise NotImplementedError(
                "Object class " + object_class + " is not supported."
            )
        else:
            self.parameter = self.object_class_parameters[object_class]
