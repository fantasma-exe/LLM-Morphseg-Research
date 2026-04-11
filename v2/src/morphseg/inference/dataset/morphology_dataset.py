from torch.utils.data import Dataset


class MorphologyInferenceDataset(Dataset):
    """
    Dataset that generates prompts for inference from a list of words.

    Instead of tokenizing each word individually, this dataset returns
    fully formatted prompts. Tokenization is then performed in batches
    inside the predictor, which is typically more efficient.

    Parameters
    ----------
    words : list of str
        A list of input words used to construct prompts.

    prompt_template : str
        A template string used to generate prompts. The template must
        contain a placeholder for the word (e.g., `"Translate: {}"`).
    """

    def __init__(self, words: list[str], prompt_template: str) -> None:
        self.words = words
        self.prompt_template = prompt_template

    def __getitem__(self, index: int) -> dict[str, str]:
        """
        Return a formatted prompt for the given index.

        Parameters
        ----------
        index : int
            Index of the word in the dataset.

        Returns
        -------
        dict[str, str]
            {"word": ..., "prompt": ...}.
        """

        word = self.words[index]
        prompt = self.prompt_template.format(word=word)
        return {"word": word, "prompt": prompt}

    def __len__(self) -> int:
        """
        Return the total number of words in the dataset.

        Returns
        -------
        int
            Number of elements in the dataset.
        """

        return len(self.words)
