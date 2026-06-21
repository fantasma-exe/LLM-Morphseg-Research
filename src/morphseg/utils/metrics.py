import numpy as np

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
    ) -> tuple[list[tuple[int, int, str]], bool]:
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
        stats = {m_type: [0, 0, 0] for m_type in self.allowed_types}
        stats["_all_"] = [0, 0, 0]

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

            word_len = len(word)
            if word_len > 0:
                total_char_acc += self._calculate_char_accuracy(
                    p_spans, t_spans, word_len
                )

            self._update_span_stats(p_spans, t_spans, stats)

        results = {}
        for m_type, (tp, fp, fn) in stats.items():
            p, r, f1 = self._calculate_f1_from_counts(tp, fp, fn)
            suffix = "full" if m_type == "_all_" else m_type.lower()
            results[f"morpheme_f1_{suffix}"] = f1
            if m_type == "_all_":
                results["morpheme_precision_full"] = p
                results["morpheme_recall_full"] = r

        n = len(words) if words else 1
        results.update(
            {
                "word_accuracy": total_word_acc / n,
                "char_level_accuracy": total_char_acc / n,
                "hallucination_rate": hallucinations_count / n,
            }
        )

        return results

    def _update_span_stats(self, p_spans, t_spans, stats):
        p_set = set(p_spans)
        t_set = set(t_spans)

        tp_all = p_set & t_set
        stats["_all_"][0] += len(tp_all)
        stats["_all_"][1] += len(p_set - t_set)
        stats["_all_"][2] += len(t_set - p_set)

        for m_type in self.allowed_types:
            p_filtered = {s for s in p_set if s[2] == m_type}
            t_filtered = {s for s in t_set if s[2] == m_type}
            stats[m_type][0] += len(p_filtered & t_filtered)
            stats[m_type][1] += len(p_filtered - t_filtered)
            stats[m_type][2] += len(t_filtered - p_filtered)

    def _calculate_char_accuracy(self, p_spans, t_spans, length):
        p_mask = np.full(length, -1, dtype=int)
        t_mask = np.full(length, -1, dtype=int)

        tag_to_id = {tag: i for i, tag in enumerate(self.allowed_types)}

        for s, e, t in p_spans:
            if t in tag_to_id:
                p_mask[max(0, s) : min(length, e)] = tag_to_id[t]
        for s, e, t in t_spans:
            if t in tag_to_id:
                t_mask[max(0, s) : min(length, e)] = tag_to_id[t]

        return np.mean(p_mask == t_mask)

    def _calculate_f1_from_counts(self, tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1
