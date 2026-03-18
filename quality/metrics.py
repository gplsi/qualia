from abc import ABC, abstractmethod
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
