def normalize_model_name(name: str) -> str:
    """
    Normalize HuggingFace model name to be filesystem-friendly.

    Replaces slashes in model names (e.g. ``microsoft/Phi-3-mini``)
    with underscores so the name can be safely used as a directory name.

    Parameters
    ----------
    name : str
        Original model name from the configuration.

    Returns
    -------
    str
        Normalized model name suitable for filesystem paths.
    """
    return name.replace("/", "_")
