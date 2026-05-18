import tiktoken

from statistics import mean, stdev
from typing import Iterable, Any, List

from quality.metrics.base import BaseMetric


class TokenCountMetric(BaseMetric):
    """Counts approximate LLM tokens in text fields.

    Uses OpenAI's BPE tokenizer via `tiktoken`. This is a closer match
    to how many LLMs count tokens than simple whitespace splitting.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:

        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return "token-count"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def compute(self, values: Iterable[Any]) -> Any:
        total_tokens = 0
        tokens_per_document = []
        for v in values:
            if v is None:
                continue
            token_count = self._count_tokens(str(v))
            total_tokens += token_count
            tokens_per_document.append(token_count)

        mean_tokens = mean(tokens_per_document) if tokens_per_document else 0
        stdev_tokens = stdev(tokens_per_document) if len(tokens_per_document) > 1 else 0
        min_tokens = min(tokens_per_document) if tokens_per_document else 0
        max_tokens = max(tokens_per_document) if tokens_per_document else 0

        return {
            "total_tokens": total_tokens,
            "tokens_per_document": {
                "mean": mean_tokens,
                "stdev": stdev_tokens,
                "min": min_tokens,
                "max": max_tokens,
            },
        }


class VocabularySizeMetric(BaseMetric):
    """Counts the number of distinct whitespace-delimited tokens in text fields."""

    @property
    def name(self) -> str:
        return "vocabulary-size"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        vocab = set()
        for v in values:
            if v is None:
                continue
            vocab.update(str(v).split())
        return len(vocab)
