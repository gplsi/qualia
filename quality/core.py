import json
from typing import Iterable, Any, Set, Dict, Tuple

from quality.metrics import BaseMetric


# ==========================================
# Core helpers
# ==========================================


def lazy_value_generator(filepath: str, key: str) -> Iterable[Any]:
    """Yields values for a specific key lazily from the JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if key in record:
                yield record[key]


def process_metric_task(
    metric: BaseMetric, filepath: str, key: str
) -> Tuple[str, str, Any]:
    """Task executed by parallel worker processes."""
    # The generator is instantiated INSIDE the worker to avoid pickling issues
    lazy_stream = lazy_value_generator(filepath, key)
    result = metric.compute(lazy_stream)
    return metric.name, key, result


def get_all_keys_from_file(filepath: str) -> Set[str]:
    """Peeks at the first line of the JSONL to discover available keys."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return set(json.loads(line).keys())
    return set()
