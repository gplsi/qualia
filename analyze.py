import argparse
import json
import os
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Iterable, Any, List, Set, Dict

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


# ==========================================
# 3. Core System & Parallel Execution
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
) -> tuple[str, str, Any]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL files using an extensible framework."
    )
    parser.add_argument("input_file", help="Path to the JSONL input file")
    parser.add_argument(
        "--text-keys", action="append", default=[], help="Keys categorized as text"
    )
    parser.add_argument(
        "--metadata-keys",
        action="append",
        default=[],
        help="Keys categorized as metadata",
    )
    args = parser.parse_args()

    # Determine key categorizations
    all_file_keys = get_all_keys_from_file(args.input_file)
    categorized_keys = set(args.text_keys + args.metadata_keys)

    key_types = {}
    for k in args.text_keys:
        key_types[k] = "text"
    for k in args.metadata_keys:
        key_types[k] = "metadata"
    for k in all_file_keys - categorized_keys:
        key_types[k] = "other"

    # Register metrics
    registered_metrics = [
        HelloMetric(),
        LengthMetric(),
        AlphanumericMetric(),
        DateInMetadataMetric(),
    ]

    # Map metrics to their target keys
    tasks = []
    for metric in registered_metrics:
        for key, ktype in key_types.items():
            if ktype in metric.target_types or "all" in metric.target_types:
                tasks.append((metric, key))

    # Execute tasks in parallel
    results_dict: Dict[str, Dict[str, Any]] = {}

    # Using ProcessPoolExecutor to bypass the GIL for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_metric_task, metric, args.input_file, key)
            for metric, key in tasks
        ]

        for future in concurrent.futures.as_completed(futures):
            metric_name, key, result = future.result()
            if metric_name not in results_dict:
                results_dict[metric_name] = {}
            results_dict[metric_name][key] = result

    # Format the final output structure matching the order of registered metrics
    final_output = {
        metric.name: results_dict.get(metric.name, {}) for metric in registered_metrics
    }

    # Determine output file name (.stats.json suffix)
    base_path, ext = os.path.splitext(args.input_file)
    output_filename = f"{base_path}.stats.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Analysis complete. Results written to {output_filename}")


if __name__ == "__main__":
    main()
