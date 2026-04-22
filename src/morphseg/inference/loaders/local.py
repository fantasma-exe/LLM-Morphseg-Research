import os

from .base import ArtifactLoader


class LocalCheckpointLoader(ArtifactLoader):
    """
    Artifact loader that retrieves a checkpoint from the local filesystem.

    This loader is intended for local development and testing scenarios where
    checkpoints are already available on disk. It simply validates the provided
    path and returns it without performing any download or artifact resolution.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file on the local filesystem.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def download_checkpoint(self) -> str:
        """
        Return the path to the local checkpoint file.

        The method verifies that the checkpoint file exists and returns its path.

        Returns
        -------
        str
            Path to the checkpoint file on the local filesystem.

        Raises
        ------
        FileNotFoundError
            If the specified checkpoint file does not exist.
        """

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        return self.checkpoint_path
