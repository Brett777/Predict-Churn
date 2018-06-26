import os
import io
import sys
import dill
import six
from datetime import datetime

from .evaluator import Evaluator
from .utils import (
    post_to_platform,
    get_current_notebook,
    strip_output,
    get_current_notebook,
    mkdir_p,
    to_list,
)


class DataScienceFramework(object):
    def __init__(
        self,
        model,
        problem_class,
        x_test,
        y_test,
        name=None,
        description=None,
        evaluator=Evaluator,
    ):
        # assign variables to class
        self.name = name
        self.description = description
        self.model = model
        self.problem_class = problem_class
        self.y_test = to_list(y_test, flatten=True)
        self.x_test = to_list(x_test)
        self.framework = model.__module__.split(".")[0]

        # get environment data
        self._meta_data = self.meta_data()

        self.y_pred = self.predict()
        # initialize evaluator
        self.evaluator = Evaluator(self.problem_class)

        # initialize saving response
        self.response = None

    # class methods
    @classmethod
    def load(cls, model_id):
        # use hard coded string to load for now
        with open(".model_cache/sklearn_model_cache.pkl", "rb") as file:
            instance = dill.load(file)
            instance.model = instance.parse_model(io.BytesIO(instance.model_serialized))
        return instance

    @classmethod
    def project_models(cls):
        query = """
            query($service_name: String!) {
                runnableInstance(serviceName: $service_name) {
                    runnable {
                        project {
                            name
                            models {
                                edges {
                                    node {
                                        id
                                        name 
                                        description
                                        problemClass
                                        framework
                                        objectClass
                                        language
                                        languageVersion
                                        createdAt
                                        updatedAt
                                        rank
                                        hyperParameters
                                        structure
                                        author {
                                            fullName
                                        }
                                        metrics {
                                            edges {
                                                node {
                                                    key
                                                    value
                                                }
                                            }
                                        }
                                        diagnostics {
                                            edges {
                                                node {
                                                    ... on ModelDiagnosticROC {
                                                    title
                                                    falsePositiveRates
                                                    truePositiveRates
                                                    thresholds
                                                    }
                                                    ... on ModelDiagnosticResidual {
                                                    title
                                                    observations
                                                    residuals
                                                    }
                                                    ... on ModelDiagnosticConfusionMatrix {
                                                    title
                                                    matrix
                                                    }
                                                }
                                            }
                                        }
                                        parameters {
                                            edges {
                                                node {
                                                    key
                                                    value
                                                    confidenceInterval {
                                                    positive
                                                    negative
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """

        response = post_to_platform(
            {"query": query, "variables": {"service_name": os.environ["SERVICE_NAME"]}}
        )
        response_data = response.json()["data"]
        models = list(
            map(
                lambda edge: edge["node"],
                response_data["runnableInstance"]["runnable"]["project"]["models"][
                    "edges"
                ],
            )
        )
        return models

    # framework dependent functions
    def predict(self):
        """ Make prediction based on x_test """
        raise NotImplementedError

    def framework_version(self):
        """ Return version of the framework been used. """
        raise NotImplementedError

    def object_class(self):
        """ Return name of the model object.  """
        raise NotImplementedError

    def parameter(self):
        """ Get parameter from model. """
        raise NotImplementedError

    def hyperparameter(self):
        """ Get hyper parameter from model. """
        raise NotImplementedError

    def serialize_model(self):
        """ Default methods for serialize model. """
        return dill.dumps(self.model)

    def parse_model(self, model_file):
        """ Default methods for reading in model. """
        return dill.load(model_file)

    # base framework functions
    def meta_data(self):
        """ Capture environment meta data. """
        meta_data_obj = {
            "name": self.name,
            "description": self.description,
            "framework": self.framework,
            "createdAt": datetime.now().isoformat(),
            "sessionName": os.environ["SERVICE_NAME"],
            "language": "python",
            "languageVersion": ".".join(map(str, sys.version_info[0:3])),
        }
        return meta_data_obj

    def diagnostics(self):
        """ Return diagnostics of model. """
        return [fn(self.y_test, self.y_pred) for fn in self.evaluator.diagnostics]

    def metrics(self):
        """ Return evaluation of model performance. """
        return [fn(self.y_test, self.y_pred) for fn in self.evaluator.metrics]

    def summary(self):
        """ Return all infomation that will be stored. """
        model_meta = {
            "diagnostics": self.diagnostics(),
            "metrics": self.metrics(),
            "parameters": self.parameter(),
            "frameworkVersion": self.framework_version(),
            "hyperParameters": self.hyperparameter(),
            "problemClass": self.problem_class,
            "objectClass": self.object_class(),
        }

        model_meta.update(self._meta_data)
        return model_meta

    def save(self):
        """ Save all information to platform. """
        if self.response:
            six.print_("Model already saved to platform.")
        else:
            self.model_serialized = self.serialize_model()

            # save model object locally for now
            mkdir_p(".model_cache")
            with open(".model_cache/sklearn_model_cache.pkl", "w") as file:
                dill.dump(self, file)

            model_meta = self.summary()

            model_meta.update(
                {
                    "data": {"y_pred": list(self.y_pred), "y_test": list(self.y_test)},
                    "notebook": get_current_notebook(),
                }
            )

            query = """
                mutation($input: CreateModelInput!) {
                    createModel(input: $input) {
                        clientMutationId
                    }
                }
                """

            self.response = post_to_platform(
                {"query": query, "variables": {"input": model_meta}}
            )
            six.print_("Model saved to platform.")
