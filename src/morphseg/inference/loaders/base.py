from abc import ABC, abstractmethod


class ArtifactLoader(ABC):
    @abstractmethod
    def download_checkpoint(self) -> str:
        """
        Download a model checkpoint artifact associated with a training run.

        This method defines a common interface for retrieving checkpoints from
        different artifact storage backends (e.g., local filesystem, MLflow,
        object storage). Implementations are responsible for locating the
        requested artifact, downloading it if necessary, and returning the
        path to the local file.

        Returns
        -------
        str
            Path to the checkpoint file on the local filesystem.
        """
        pass
