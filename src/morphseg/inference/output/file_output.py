from pathlib import Path

from .base import BaseOutput


class FileOutput(BaseOutput):
    """
    Output implementation that writes inference results to a file.
    """

    def __init__(self, output_path: str) -> None:
        """
        Initialize the file output.

        Parameters
        ----------
        output_path : str
            Path to the file where the results will be saved.
        """

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, inputs: list[str], predictions: list[str]) -> None:
        """
        Write input strings and their corresponding predictions to a file.

        Parameters
        ----------
        inputs : list[str]
            The original input strings.

        predictions : list[str]
            The predicted strings corresponding to each input.
        """

        with open(self.output_path, "w", encoding="utf-8") as f:
            for input, pred in zip(inputs, predictions):
                f.write(f"{input} -> {pred}")
        print(f"Resulsts saved in {self.output_path}")
