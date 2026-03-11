from base import BaseOutput


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
        self.output_path = output_path

    def write(self, inputs: list[str], predictions: list[str]) -> None:
        """
        Write input strings and their corresponding predictions to a file.

        Parameters
        ----------
        inputs : list of str
            The original input strings.

        predictions : list of str
            The predicted strings corresponding to each input.
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for word, pred in zip(inputs, predictions):
                f.write(f"{word} -> {pred}")
        print(f"Resulsts saved in {self.output_path}")
