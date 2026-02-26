def build_prompt(word: str, answer: str | None = None) -> str:
    """
    Build a prompt for morphological segmentation.

    The prompt instructs the model to split the given word into morphemes
    and specify the type of each morpheme. If an answer is provided, it is
    appended to the prompt (useful for supervised fine-tuning).

    Parameters
    ----------
    word : str
        Input word to be segmented.
    answer : str or None, optional
        Reference answer to append to the prompt.

    Returns
    -------
    str
        Formatted prompt string.
    """
    prompt = f"""
    ### Инструкция:
    Раздели слово на морфемы и укажи тип каждой морфемы

    ### Слово:
    {word}

    ### Ответ:

    """

    if answer is not None:
        prompt += answer

    return prompt
