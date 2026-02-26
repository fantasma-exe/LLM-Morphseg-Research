from typing import Callable

from metrics import morpheme_precision, morpheme_recall, morpheme_f1, char_accuracy, word_accuracy

METRICS: dict[str, Callable[[str, str], float]] = {
    "morpheme_precision_full": lambda p, g: morpheme_precision(p, g),
    "morpheme_recall_full": lambda p, g: morpheme_recall(p, g),
    "morpheme_f1_full": lambda p, g: morpheme_f1(p, g),
    "morpheme_precision_root": lambda p, g: morpheme_precision(p, g, allowed_types={"ROOT"}),
    "morpheme_recall_root": lambda p, g: morpheme_recall(p, g, allowed_types={"ROOT"}),
    "morpheme_f1_root": lambda p, g: morpheme_f1(p, g, allowed_types={"ROOT"}),
    "char_level_accuracy": char_accuracy,
    "word_accuracy": word_accuracy,
}
