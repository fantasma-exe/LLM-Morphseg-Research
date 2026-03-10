from base import BaseOutput


class ConsoleOutput(BaseOutput):
    """
    Output implementation that prints inference results to the console.
    """

    def write(self, inputs: list[str], predictions: list[str]) -> None:
        """
        Print input strings and their corresponding predictions to the console.

        Parameters
        ----------
        inputs : list of str
            The original input strings.
        predictions : list of str
            The predicted strings corresponding to each input.
        """
        print("\n--- Inference results ---")
        for word, pred in zip(inputs, predictions):
            print(f"{word} -> {pred}")
        print("---------------------------")
