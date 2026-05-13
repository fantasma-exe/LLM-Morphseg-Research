from collections import defaultdict
from typing import Any


class MorphemeMetrics:
    def __init__(self, allowed_types: set[str] | None = None):
        self.allowed_types = allowed_types or {
            "ROOT",
            "PREF",
            "SUFF",
            "END",
            "LINK",
            "POST",
            "HYPN",
        }

    def _parse_to_spans(
        self, prediction: str, original_word: str
    ) -> tuple[list[tuple[int, int, list[str]]], bool]:
        if not prediction or not isinstance(prediction, str):
            return [], False

        parts = prediction.split("/")
        spans = []
        current_idx = 0
        reconstructed = ""

        for part in parts:
            if ":" not in part:
                continue
            morpheme, tag = part.rsplit(":", 1)
            start = current_idx
            end = current_idx + len(morpheme)
            spans.append((start, end, tag))
            current_idx = end
            reconstructed += morpheme

        is_hallucination = reconstructed != original_word
        return spans, is_hallucination

    def compute(
        self, preds: list[str], targets: list[str], words: list[str]
    ) -> dict[str, Any]:
        stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        total_word_acc = 0
        total_char_acc = 0
        hallucinations_count = 0

        for p_str, t_str, word in zip(preds, targets, words):
            p_spans, is_halluc = self._parse_to_spans(p_str, word)
            t_spans, _ = self._parse_to_spans(t_str, word)

            if is_halluc:
                hallucinations_count += 1

            if p_str == t_str:
                total_word_acc += 1

            total_char_acc += self._calculate_char_accuracy(p_spans, t_spans, len(word))

            self._update_span_stats(p_spans, t_spans, stats)

        results = {}

        full_p, full_r, full_f1 = self._get_f1(stats["_all_"])
        results.update(
            {
                "morpheme_precision_full": full_p,
                "morpheme_recall_full": full_r,
                "morpheme_f1_full": full_f1,
            }
        )

        for m_type in self.allowed_types:
            p, r, f1 = self._get_f1(stats[m_type])
            results[f"morpheme_f1_{m_type.lower()}"] = f1

        n = len(words) if words else 1
        results["word_accuracy"] = total_word_acc / n
        results["char_level_accuracy"] = total_char_acc / n
        results["hallucination_rate"] = hallucinations_count / n

        return results

    def _update_span_stats(self, p_spans, t_spans, stats):
        p_set = set(p_spans)
        t_set = set(t_spans)

        stats["_all_"]["tp"] += len(p_set & t_set)
        stats["_all_"]["fp"] += len(p_set - t_set)
        stats["_all_"]["fn"] += len(t_set - p_set)

        for m_type in self.allowed_types:
            p_filtered = {s for s in p_set if s[2] == m_type}
            t_filtered = {s for s in t_set if s[2] == m_type}
            stats[m_type]["tp"] += len(p_filtered & t_filtered)
            stats[m_type]["fp"] += len(p_filtered - t_filtered)
            stats[m_type]["fn"] += len(t_filtered - p_filtered)

    def _calculate_char_accuracy(self, p_spans, t_spans, length):
        if length == 0:
            return 0

        p_mask = [""] * length
        t_mask = [""] * length

        for s, e, t in p_spans:
            p_mask[max(0, s) : min(length, e)] = [t] * (min(length, e) - max(0, s))
        for s, e, t in t_spans:
            t_mask[max(0, s) : min(length, e)] = [t] * (min(length, e) - max(0, s))

        correct = sum(1 for p, t in zip(p_mask, t_mask) if p == t)

        return correct / length

    def _get_f1(self, s):
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1
