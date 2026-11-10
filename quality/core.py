import json
from typing import Iterable, Any, Set, Dict, Tuple

from quality.metrics import BaseMetric

# ==========================================
# Core helpers
# ==========================================


def lazy_value_generator(
    filepath: str,
    key: str,
    language_key: str | None = None,
    language: str | None = None,
) -> Iterable[Any]:
    """Yields values for a key, optionally filtering records by language."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if language is not None:
                record_language = record.get(language_key) if language_key else None
                if (
                    not isinstance(record_language, str)
                    or record_language.strip() != language
                ):
                    continue
            if key in record:
                yield record[key]


def process_metric_task(
    metric: BaseMetric,
    filepath: str,
    key: str,
    language_key: str | None = None,
    language: str | None = None,
) -> Tuple[str, str, str | None, Any]:
    """Task executed by parallel worker processes."""
    # The generator is instantiated INSIDE the worker to avoid pickling issues
    lazy_stream = lazy_value_generator(filepath, key, language_key, language)
    result = metric.compute(lazy_stream)
    return metric.name, key, language, result


def get_all_languages_from_file(filepath: str, language_key: str) -> list[str]:
    """Return sorted, non-empty language values found in a JSONL file."""
    languages = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            language = record.get(language_key)
            if isinstance(language, str) and language.strip():
                languages.add(language.strip())
    return sorted(languages)


def get_all_keys_from_file(filepath: str) -> Set[str]:
    """Peeks at the first line of the JSONL to discover available keys."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return set(json.loads(line).keys())
    return set()
