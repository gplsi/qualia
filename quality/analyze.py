import argparse
import json
import os
import concurrent.futures
from typing import Any, Dict
import datasets

from quality.metrics import (
    HelloMetric,
    LengthMetric,
    AlphanumericMetric,
    MetadataKeywordGroupsMetric,
    VACARatioMetric,
    TokenCountMetric,
    VocabularySizeMetric,
    StochasticDiversityMetric,
    InnerOuterDiversityMetric,
    MinhashDuplicationMetric,
    MetadataYearCountMetric,
    GenderBiasMetric,
)
from quality.core import process_metric_task, get_all_keys_from_file


def fetch_huggingface_dataset(dataset_name: str, cache_dir: str) -> str:
    """Fetch a Hugging Face dataset and return the local path."""
    dataset = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    local_path = os.path.join(cache_dir, dataset_name)
    os.makedirs(local_path, exist_ok=True)
    dataset.to_json(os.path.join(local_path, "data.jsonl"))
    return os.path.join(local_path, "data.jsonl")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSONL files using an extensible framework."
    )
    parser.add_argument("input_file", help="Path to the JSONL input file or Hugging Face dataset name")
    parser.add_argument(
        "--input-type",
        choices=["local", "huggingface"],
        default="local",
        help="Specify the input type (default: local path)",
    )
    parser.add_argument(
        "--text-keys", action="append", default=[], help="Keys categorized as text"
    )
    parser.add_argument(
        "--metadata-keys",
        action="append",
        default=[],
        help="Keys categorized as metadata",
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="Enable develop mode (exceptions are not handled)",
    )
    args = parser.parse_args()

    # Handle input type
    if args.input_type == "huggingface":
        cache_dir = os.path.join("data", ".cache")
        args.input_file = fetch_huggingface_dataset(args.input_file, cache_dir)

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
        MetadataKeywordGroupsMetric(),
        VACARatioMetric(),
        TokenCountMetric(),
        VocabularySizeMetric(),
        StochasticDiversityMetric(),
        InnerOuterDiversityMetric(),
        MinhashDuplicationMetric(),
        MetadataYearCountMetric(),
        GenderBiasMetric(),
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
        futures = {
            executor.submit(process_metric_task, metric, args.input_file, key): (
                metric,
                key,
            )
            for metric, key in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            metric, key = futures[future]
            metric_name = getattr(metric, "name", "unknown_metric")

            try:
                metric_name, key, result = future.result()
                if metric_name not in results_dict:
                    results_dict[metric_name] = {}
                results_dict[metric_name][key] = result
            except Exception as e:
                # On failure, record the error instead of breaking
                if metric_name not in results_dict:
                    results_dict[metric_name] = {}
                key_for_error = key or "unknown_job"

                if args.develop:
                    # In develop mode, let exceptions propagate
                    raise RuntimeError(
                        f"Error processing metric '{metric_name}' for key '{key_for_error}': {e}"
                    ) from e

                results_dict[metric_name][key_for_error] = {"error": str(e)}

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
