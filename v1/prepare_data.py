import json

from pathlib import Path


def convert_data(input_path: str, output_path: str) -> None:
    """
    Convert raw morphological data to JSONL format.

    The input file is expected to contain one example per line with
    a word and its target segmentation separated by whitespace.
    The output is written in JSON Lines format with fields ``input``
    and ``output``.

    Parameters
    ----------
    input_path : str
        Path to the input text file.
    output_path : str
        Path to the output JSONL file.

    Returns
    -------
    None
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path, "r", encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            word, target = line.split()
            sample = {
                "input": word,
                "output": target,
            }
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")


convert_data("tikhonov/morphodict-t-train.txt", "data/train.jsonl")
convert_data("tikhonov/morphodict-t-test.txt", "data/test.jsonl")
