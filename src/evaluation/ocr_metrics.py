"""OCR metrics: character error rate, word error rate, detection success.

Standard Levenshtein-distance based metrics:

* CER  = edit distance (characters) / reference character count.
* WER  = edit distance (words) / reference word count.
* detection success = fraction of reference strings that produced at
  least one recognised line with >= 50% of its words correct.
"""
from typing import Sequence


def _levenshtein(a: str, b: str) -> int:
    """Classic DP edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER between two strings (0.0 = perfect)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER between two strings, tokenised on whitespace."""
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not hypothesis.split() else 1.0
    return _levenshtein(reference, hypothesis) / len(ref_words)


def text_detection_success(
    references: Sequence[str],
    recognised: Sequence[str],
    correct_word_fraction: float = 0.5,
) -> float:
    """Fraction of references 'successfully' detected.

    A reference is a success if at least one recognised string shares
    ``correct_word_fraction`` of its words (case-insensitive) OR its WER
    is below ``1 - correct_word_fraction``.

    Args:
        references: Ground-truth text strings.
        recognised: OCR output strings.
        correct_word_fraction: Word overlap threshold for a hit.

    Returns:
        Detection success rate in [0, 1].
    """
    if not references:
        return 1.0
    hits = 0
    for ref in references:
        ref_words = set(ref.lower().split())
        for hyp in recognised:
            hyp_words = set(hyp.lower().split())
            if not ref_words:
                hits += 1
                break
            overlap = len(ref_words & hyp_words) / len(ref_words)
            if overlap >= correct_word_fraction:
                hits += 1
                break
    return hits / len(references)