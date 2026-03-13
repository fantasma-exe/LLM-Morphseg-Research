import mlflow

from mlflow.client import MlflowClient

from .base import ArtifactLoader


class MLflowLoader(ArtifactLoader):
    """
    Artifact loader that retrieves model checkpoints from MLflow artifact storage.

    The loader connects to an MLflow tracking server, accesses a specific run,
    reads the artifact path stored in the ``best_checkpoint`` run tag, and
    downloads the corresponding checkpoint to a local directory.

    Parameters
    ----------
    tracking_uri : str
        URI of the MLflow tracking server (e.g., local path, HTTP endpoint,
        or remote artifact store configuration).

    run_id : str
        Identifier of the MLflow run from which the checkpoint should be
        retrieved.

    dst_dir : str
        Local directory where the checkpoint artifact will be downloaded.
        The directory will be created if it does not exist.
    """

    def __init__(self, tracking_uri: str, run_id: str, dst_dir: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.run_id = run_id
        self.dst_dir = dst_dir

    def download_checkpoint(
        self,
    ) -> str:
        """
        Download the best checkpoint associated with the configured MLflow run.

        The method queries MLflow for the run specified during initialization,
        retrieves the artifact path stored in the ``best_checkpoint`` tag,
        and downloads that artifact to the local destination directory.

        Returns
        -------
        str
            Path to the downloaded checkpoint file on the local filesystem.
        """

        run = self.client.get_run(self.run_id)

        artifact_path = run.data.tags["best_checkpoint"]

        return self.client.download_artifacts(
            self.run_id, artifact_path, dst_path=self.dst_dir
        )
