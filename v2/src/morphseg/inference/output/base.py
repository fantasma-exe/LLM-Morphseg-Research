from abc import ABC, abstractmethod


class BaseOutput(ABC):
    @abstractmethod
    def write(self, inputs: list[str], predictions: list[str]) -> None:
        """
        Save or display prediction results.

        Parameters
        ----------
        inputs : list of str
            The original input strings.

        predictions : list of str
            The predicted strings corresponding to each input.
        """
        pass
