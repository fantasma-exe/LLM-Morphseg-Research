from abc import ABC, abstractmethod


class BaseInput(ABC):
    @abstractmethod
    def read(self) -> list[str]:
        """
        Read input data and return a list of strings.

        Returns
        -------
        list[str]
            A list containing the read strings (words).
        """
        pass
