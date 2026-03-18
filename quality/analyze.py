import argparse
import json
import os
import concurrent.futures
from typing import Any, Dict

from quality.metrics import (
    HelloMetric,
    LengthMetric,
    AlphanumericMetric,
    DateInMetadataMetric,
    VACARatioMetric,
    TokenCountMetric,
    VocabularySizeMetric,
)
from quality.core import process_metric_task, get_all_keys_from_file


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
        VACARatioMetric(),
        TokenCountMetric(),
        VocabularySizeMetric(),
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
