import math
import unicodedata


def ranking_metrics(retrieved: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    relevant = {key for key, value in relevance.items() if value > 0}
    seen, gains, first = set(), [], 0
    for rank, key in enumerate(retrieved[: max(0, k)], 1):
        grade = relevance.get(key, 0) if key not in seen else 0
        if grade > 0 and not first:
            first = rank
        gains.append((2**grade - 1) / math.log2(rank + 1))
        seen.add(key)
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[: max(0, k)], 1)
    )
    return {
        "recall": len(seen & relevant) / len(relevant) if relevant else 0.0,
        "hit": float(bool(seen & relevant)),
        "mrr": 1 / first if first else 0.0,
        "ndcg": sum(gains) / ideal if ideal else 0.0,
    }


def edit_distance(left, right) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    reference = unicodedata.normalize("NFC", reference)
    prediction = unicodedata.normalize("NFC", prediction)
    return edit_distance(reference, prediction) / max(1, len(reference))


def word_error_rate(reference: str, prediction: str) -> float:
    return edit_distance(reference.split(), prediction.split()) / max(1, len(reference.split()))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))]
