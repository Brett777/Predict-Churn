import sklearn
from .datascience_framework import DataScienceFramework

from .sklearn_object_class import SklearnObjectClass


class SklearnModel(DataScienceFramework):
    def __init__(self, model, problem_class, x_test, y_test, name=None, **kwags):
        super(SklearnModel, self).__init__(
            model=model,
            problem_class=problem_class,
            x_test=x_test,
            y_test=y_test,
            name=name,
            **kwags
        )

        self.parameter = lambda: SklearnObjectClass(model.__class__.__name__).parameter(
            model
        )

    def framework_version(self):
        return sklearn.__version__

    def object_class(self):
        return self.model.__module__ + "." + self.model.__class__.__name__

    def predict(self):
        return self.model.predict(self.x_test)

    def hyperparameter(self):
        return self.model.get_params()
