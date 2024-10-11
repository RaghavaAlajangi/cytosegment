from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """Abstract base class for tracking experiments, metrics, parameters,
    and artifacts.

    This class defines an interface for logging metrics, parameters, and
    artifacts during machine learning experiments or other workflows. It is
    designed to be     extended by concrete implementations that handle
    specific logging backends (e.g., MLflow, TensorBoard, etc.).
    """

    def __init__(self, **kwargs):
        """Initialize the BaseTracker.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for configuring the tracker.
        """
        pass

    @abstractmethod
    def log_metric(self, name, metric):
        """Log a metric value under a given name.

        Parameters
        ----------
        name : str
            The name of the metric to log.
        metric : any
            The value of the metric.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def log_model(self, *args):
        """Log a trained model checkpoint.

        Parameters
        ----------
        args

        Returns
        -------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def log_param(self, name, params):
        """Log a parameter under a given name.

        Parameters
        ----------
        name : str
            The name of the parameter.
        params : str, int, float, list, or dict
            The value(s) of the parameter to log.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def log_artifact(self, artifact_name, artifact_path):
        """Log an artifact (such as a file or directory) under a given name.

        Parameters
        ----------
        artifact_name : str
            The name under which to log the artifact.
        artifact_path : str
            The path to the artifact (file or directory) to be logged.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        pass

    @abstractmethod
    def end_run(self):
        """End the current run and finalize the tracking session.

        Ensures that all metrics, parameters, and artifacts have been logged
        and the run is properly closed.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        pass
