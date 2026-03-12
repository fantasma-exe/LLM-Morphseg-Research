from typing import Iterable


def _parse(seg: str) -> list[str]:
    """
    Split a segmented word into morpheme units.

    Parameters
    ----------
    seg : str
        Morphologically segmented string with '/' as separator.

    Returns
    -------
    list[str]
        List of morpheme strings.
    """
    return [ss for s in seg.split("/") if (ss := s.strip())]


def _flatten_to_char_tags(seg: str) -> list[tuple[str, str]]:
    """
    Parse a string in the format 'morph:TYPE/morph:TYPE' into a list of (character, type) pairs.

    Parameters
    ----------
    seg : str
        A string representing the segmentation, where each token is in the form 'word:TAG',
        and tokens are separated by slashes.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples, each containing a character from the token and its associated tag.

    Examples
    --------
    'миг:ROOT' -> [('м', 'ROOT'), ('и', 'ROOT'), ('г', 'ROOT')]
    """
    result = []
    parts = [p.strip() for p in seg.split("/") if p.strip()]

    for part in parts:
        if ":" in part:
            token, tag = part.rsplit(":", 1)
        else:
            token = part
            tag = "UNKNOWN"

        for char in token:
            result.append((char, tag))

    return result


def char_accuracy(preds: Iterable[str], golds: Iterable[str]) -> float:
    """
    Compute character-level accuracy.

    A character-level accuracy score is calculated by comparing each character and its associated
    tag between predicted and gold segmentations. The score is the fraction of characters with
    correct tags and correct character predictions.

    Parameters
    ----------
    preds : Iterable[str]
        An iterable of predicted segmentations.
    golds : Iterable[str]
        An iterable of gold segmentations.

    Returns
    -------
    float
        The fraction of correctly predicted characters with the correct tags.

    Notes
    -----
    A correct character prediction requires both the character and its tag to match the gold
    segmentation. If there are multiple characters or tokens, they will be compared individually.

    Examples
    --------
    preds = ['миг:ROOT']
    golds = ['миг:ROOT']
    char_accuracy(preds, golds) -> 1.0
    """
    correct_chars = 0
    total_chars = 0

    for p_str, g_str in zip(preds, golds):
        p_seq = _flatten_to_char_tags(p_str)
        g_seq = _flatten_to_char_tags(g_str)

        max_len = max(len(p_seq), len(g_seq))

        for i in range(max_len):
            if i < len(p_seq) and i < len(g_seq):
                p_char, p_tag = p_seq[i]
                g_char, g_tag = g_seq[i]

                if p_char == g_char and p_tag == g_tag:
                    correct_chars += 1

            total_chars += 1

    return correct_chars / total_chars if total_chars > 0 else 0.0


def word_accuracy(preds: Iterable[str], golds: Iterable[str]) -> float:
    """
    Compute word-level accuracy.

    A prediction is considered correct if the entire predicted
    segmentation exactly matches the gold segmentation.

    Parameters
    ----------
    preds : Iterable[str]
        Predicted segmentations.
    golds : Iterable[str]
        Gold segmentations.

    Returns
    -------
    float
        Fraction of exactly matched words.
    """
    preds = list(preds)
    golds = list(golds)
    assert len(preds) == len(golds)

    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / len(golds) if golds else 0.0


def _filter_morphemes(
    morphemes: Iterable[str],
    allowed_types: set[str] | None,
) -> set[str]:
    """
    Filter morphemes by their type.

    Morpheme type is inferred from the suffix after ':'.

    Parameters
    ----------
    morphemes : Iterable[str]
        Morpheme strings in the form 'form:TYPE'.
    allowed_types : set[str] | None
        Set of allowed morpheme types. If None, all morphemes are kept.

    Returns
    -------
    set[str]
        Filtered set of morphemes.
    """
    result = set()
    for m in morphemes:
        if ":" not in m:
            continue
        _, m_type = m.rsplit(":", 1)
        if allowed_types is not None:
            if m_type in allowed_types:
                result.add(m)
        else:
            result.add(m)
    return result


def _morpheme_stats_single(
    pred: str,
    gold: str,
    *,
    allowed_types: set[str] | None = None,
) -> tuple[int, int, int]:
    """
    Compute morpheme-level true positives, false positives and false negatives
    for a single prediction-gold pair.

    Parameters
    ----------
    pred : str
        Predicted segmentation.
    gold : str
        Gold segmentation.
    allowed_types : set[str] | None, optional
        Morpheme types to include in evaluation.

    Returns
    -------
    tuple[int, int, int]
        (tp, fp, fn) counts for the given example.
    """
    p = _filter_morphemes(_parse(pred), allowed_types)
    g = _filter_morphemes(_parse(gold), allowed_types)

    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)

    return tp, fp, fn


def _morpheme_stats(
    preds: Iterable[str],
    golds: Iterable[str],
    *,
    allowed_types: set[str] | None = None,
) -> tuple[int, int, int]:
    """
    Aggregate morpheme-level true positives, false positives and false negatives
    over a dataset (micro-averaging).

    Parameters
    ----------
    preds : Iterable[str]
        Predicted segmentations.
    golds : Iterable[str]
        Gold segmentations.
    allowed_types : set[str] | None, optional
        Morpheme types to include in evaluation.

    Returns
    -------
    tuple[int, int, int]
        Aggregated (tp, fp, fn) counts.
    """
    tp = fp = fn = 0

    for p, g in zip(preds, golds):
        tpi, fpi, fni = _morpheme_stats_single(p, g, allowed_types=allowed_types)
        tp += tpi
        fp += fpi
        fn += fni

    return tp, fp, fn


def morpheme_precision(
    preds: Iterable[str],
    golds: Iterable[str],
    *,
    allowed_types: set[str] | None = None,
) -> float:
    """
    Compute morpheme-level precision (micro-averaged).

    Precision = TP / (TP + FP).

    Parameters
    ----------
    preds : Iterable[str]
        Predicted segmentations.
    golds : Iterable[str]
        Gold segmentations.
    allowed_types : set[str] | None, optional
        Morpheme types to include in evaluation.

    Returns
    -------
    float
        Morpheme-level precision.
    """
    tp, fp, _ = _morpheme_stats(preds, golds, allowed_types=allowed_types)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def morpheme_recall(
    preds: Iterable[str],
    golds: Iterable[str],
    *,
    allowed_types: set[str] | None = None,
) -> float:
    """
    Compute morpheme-level recall (micro-averaged).

    Recall = TP / (TP + FN).

    Parameters
    ----------
    preds : Iterable[str]
        Predicted segmentations.
    golds : Iterable[str]
        Gold segmentations.
    allowed_types : Optional[Set[str]], optional
        Morpheme types to include in evaluation.

    Returns
    -------
    float
        Morpheme-level recall.
    """
    tp, _, fn = _morpheme_stats(preds, golds, allowed_types=allowed_types)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def morpheme_f1(
    preds: Iterable[str],
    golds: Iterable[str],
    *,
    allowed_types: set[str] | None = None,
) -> float:
    """
    Compute morpheme-level F1 score (micro-averaged).

    F1 = 2 * TP / (2 * TP + FP + FN).

    Parameters
    ----------
    preds : Iterable[str]
        Predicted segmentations.
    golds : Iterable[str]
        Gold segmentations.
    allowed_types : set[str] | None, optional
        Morpheme types to include in evaluation.

    Returns
    -------
    float
        Morpheme-level F1 score.
    """
    tp, fp, fn = _morpheme_stats(preds, golds, allowed_types=allowed_types)

    if tp == 0:
        return 0.0

    return 2 * tp / (2 * tp + fp + fn)
