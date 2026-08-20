"""OCR metrics: character error rate, word error rate, detection success.

Standard Levenshtein-distance based metrics:

* CER  = edit distance (characters) / reference character count.
* WER  = edit distance (words) / reference word count.
* detection success = fraction of reference strings that produced at
  least one recognised line with >= 50% of its words correct.
"""
from typing import Dict, Sequence


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
    """WER between two strings, tokenised on whitespace.

    Levenshtein distance is computed over *word tokens* (not characters)
    and divided by the reference word count, so e.g. a single swapped
    word in a 4-word phrase gives 0.25 (not a character-scale value).
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def exact_match(reference: str, hypothesis: str) -> int:
    """1 if the strings match exactly (case/whitespace-insensitive)."""
    ref = " ".join(reference.lower().split())
    hyp = " ".join(hypothesis.lower().split())
    return int(ref == hyp and bool(ref))


def aggregate_ocr_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    correct_word_fraction: float = 0.5,
) -> Dict[str, float]:
    """Mean CER, WER, exact-match and detection-success over paired texts.

    Args:
        references: Ground-truth texts.
        hypotheses: OCR outputs, same length/order as ``references``.
        correct_word_fraction: Overlap threshold for ``text_detection_success``.

    Returns:
        dict with "cer", "wer", "exact_match", "detection_success".
    """
    cer = [character_error_rate(r, h) for r, h in zip(references, hypotheses)]
    wer = [word_error_rate(r, h) for r, h in zip(references, hypotheses)]
    exact = [exact_match(r, h) for r, h in zip(references, hypotheses)]
    return {
        "cer": sum(cer) / len(cer) if cer else 0.0,
        "wer": sum(wer) / len(wer) if wer else 0.0,
        "exact_match": sum(exact) / len(exact) if exact else 0.0,
        "detection_success": text_detection_success(
            references, hypotheses, correct_word_fraction),
    }


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