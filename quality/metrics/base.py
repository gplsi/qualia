from abc import ABC, abstractmethod
from typing import Iterable, Any, List


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

    def for_language(self, language: str) -> "BaseMetric":
        """Return a metric configured for a language, or self if language-neutral."""
        return self

    @abstractmethod
    def compute(self, values: Iterable[Any]) -> Any:
        """Processes a lazy sequence of values and returns the aggregate result."""
        pass
