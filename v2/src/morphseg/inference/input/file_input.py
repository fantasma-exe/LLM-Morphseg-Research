import os

from .base import BaseInput


class FileInput(BaseInput):
    """
    Input source that reads words from a file.
    """

    def __init__(self, input_path: str) -> None:
        """
        Initialize the file input.

        Parameters
        ----------
        input_path : str
            Path to the input file.
        """
        self.input_path = input_path

    def read(self) -> list[str]:
        """
        Read lines from the file and return them as a list of strings.

        Returns
        -------
        list[str]
            A list of strings where each element corresponds to a stripped
            line from the file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not os.path.exists(self.input_path):
            raise FileNotFoundError()

        with open(self.input_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]
