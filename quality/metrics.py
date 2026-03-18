from abc import ABC, abstractmethod
from statistics import mean
from typing import Iterable, Any, List

# ==========================================
# 1. Extensible Framework (Base Classes)
# ==========================================


class BaseMetric(ABC):
    """Base class for all metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The key name used in the final JSON output."""
        pass

    @property
    @abstractmethod
    def target_types(self) -> List[str]:
        """Column types to process: e.g., 'text', 'metadata', 'all', 'other'."""
        pass

    @abstractmethod
    def compute(self, values: Iterable[Any]) -> Any:
        """Processes a lazy sequence of values and returns the aggregate result."""
        pass


# ==========================================
# 2. Metric Implementations
# ==========================================


class HelloMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "hello"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        # Exhaust the generator to simulate processing, then return the static value
        for _ in values:
            pass
        return "world!"


class LengthMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "length"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        return sum(len(str(v)) for v in values if v is not None)


class AlphanumericMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "alphanumeric"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        return sum(
            sum(1 for c in str(v) if c.isalnum()) for v in values if v is not None
        )


class DateInMetadataMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "with-date-in-metadata"

    @property
    def target_types(self) -> List[str]:
        return ["metadata"]

    def compute(self, values: Iterable[Any]) -> Any:
        not_empty = False
        for value in values:
            not_empty = True
            if not any(
                keyword in str(value).lower()
                for keyword in ["date", "timestamp", "datetime"]
            ):
                return False
        return not_empty


class VACARatioMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "vaca-ratio"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        from disvaca import VACADiscriminator

        discriminator = VACADiscriminator.load()

        all_counts = [
            discriminator.get_counts(value) for value in values if value is not None
        ]

        ratios = []
        all_va_count = 0
        all_ca_count = 0
        all_total_counts = 0

        for counts in all_counts:
            va_count = counts.get("va", 0)
            ca_count = counts.get("ca", 0)
            total_count = sum(counts.values())

            va_ratio = va_count / total_count if total_count > 0 else 0.0
            ca_ratio = ca_count / total_count if total_count > 0 else 0.0
            ratios.append([va_ratio, ca_ratio])

            all_va_count += va_count
            all_ca_count += ca_count
            all_total_counts += total_count

        micro_ratios = {
            "va": all_va_count / all_total_counts if all_total_counts > 0 else 0.0,
            "ca": all_ca_count / all_total_counts if all_total_counts > 0 else 0.0,
        }

        macro_ratios = {
            "va": mean(r[0] for r in ratios) if ratios else 0.0,
            "ca": mean(r[1] for r in ratios) if ratios else 0.0,
        }

        return {"micro": micro_ratios, "macro": macro_ratios}


class TokenCountMetric(BaseMetric):
    """Counts approximate LLM tokens in text fields.

    Uses OpenAI's BPE tokenizer via `tiktoken`. This is a closer match
    to how many LLMs count tokens than simple whitespace splitting.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

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
