from typing import Protocol


class TokenCounter(Protocol):
    name: str
    revision: str

    def count(self, text: str) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...


class ModelTokenCounter:
    def __init__(self, name: str, revision: str = "main", tokenizer=None):
        self.name, self.revision, self._tokenizer = name, revision, tokenizer

    def _get(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.name, revision=self.revision, use_fast=True)
        return self._tokenizer

    def count(self, text: str) -> int:
        return len(self._get()(text, add_special_tokens=False, truncation=False)["input_ids"])

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [
            tuple(pair)
            for pair in self._get()(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)[
                "offset_mapping"
            ]
            if pair[1] > pair[0]
        ]


class ConservativeTokenCounter:
    """UTF-8 byte upper bound for generation when its tokenizer is unavailable."""

    name, revision = "utf8-upper-bound", "1"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(i, i + 1) for i, char in enumerate(text) for _ in char.encode("utf-8")]
