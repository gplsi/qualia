import tiktoken

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
        for v in values:
            if v is None:
                continue
            total_tokens += self._count_tokens(str(v))
        return total_tokens


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
