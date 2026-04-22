from .base import BaseOutput


class ConsoleOutput(BaseOutput):
    """
    Output implementation that prints inference results to the console.
    """

    def write(self, inputs: list[str], predictions: list[str]) -> None:
        """
        Print input words and their corresponding predictions to the console.

        Parameters
        ----------
        inputs : list[str]
            The original input words.

        predictions : list[str]
            The predicted segmentations corresponding to each input.
        """

        print("\n--- Inference results ---")
        for input, pred in zip(inputs, predictions):
            print(f"{input} -> {pred}")
        print("---------------------------")
