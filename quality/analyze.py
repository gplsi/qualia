import argparse
import json
import os
import concurrent.futures
import logging
import datasets

from typing import Any, Dict
from tqdm import tqdm

from quality.metrics import (
    HelloMetric,
    LengthMetric,
    AlphanumericMetric,
    NumDocumentsMetric,
    MetadataKeywordGroupsMetric,
    VACARatioMetric,
    TokenCountMetric,
    VocabularySizeMetric,
    StochasticDiversityMetric,
    InnerOuterDiversityMetric,
    MinhashDuplicationMetric,
    MetadataYearCountMetric,
    ContextualGenderBiasMetric,
    GenderBiasMetric,
    STTRMetric,
    MTLDMetric,
)
from quality.core import (
    process_metric_task,
    get_all_keys_from_file,
    get_all_languages_from_file,
)

logger = logging.getLogger(__name__)


def fetch_huggingface_dataset(dataset_name: str, cache_dir: str) -> str:
    """Fetch a Hugging Face dataset and return the local path."""
    local_path = os.path.join(cache_dir, dataset_name)
    dataset_file = os.path.join(local_path, "data.jsonl")

    logger.info("Preparing Hugging Face dataset '%s' in %s", dataset_name, cache_dir)

    # Check if the dataset already exists in the cache
    if os.path.exists(dataset_file):
        logger.info("Dataset already cached at %s", dataset_file)
        return dataset_file

    logger.info("Loading dataset '%s' from Hugging Face", dataset_name)
    dataset = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    logger.info("Writing dataset '%s' to %s", dataset_name, dataset_file)
    os.makedirs(local_path, exist_ok=True)
    dataset.to_json(dataset_file)
    logger.info("Dataset '%s' cached successfully", dataset_name)
    return dataset_file


def store_result(
    results: Dict[str, Dict[str, Any]],
    mode: str,
    metric_name: str,
    key: str,
    language: str | None,
    result: Any,
) -> None:
    metric_results = results.setdefault(metric_name, {})
    is_global = mode == "global" or (mode == "both" and language is None)
    scope = "global" if is_global else language or "unspecified"
    metric_results.setdefault(key, {})[scope] = result


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    logger.info("Starting analysis")
    parser = argparse.ArgumentParser(
        description="Analyze JSONL files using an extensible framework."
    )
    parser.add_argument(
        "input_file", help="Path to the JSONL input file or Hugging Face dataset name"
    )
    parser.add_argument(
        "--input-type",
        choices=["local", "huggingface"],
        default="local",
        help="Specify the input type (default: local path)",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "language", "both"],
        default="global",
        help="Compute metrics globally, per language, or both (default: global)",
    )
    parser.add_argument(
        "--language-key",
        help="Record key containing each document's language (required for language/both mode)",
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
        "--ignore-other",
        action="store_true",
        help="Skip keys that are not explicitly listed as text or metadata.",
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="Enable develop mode (exceptions are not handled)",
    )
    parser.add_argument(
        "--update",
        action="append",
        default=[],
        help="Metric names to recompute. Requires an existing .stats.json file.",
    )
    parser.add_argument(
        "--or-insert",
        action="store_true",
        help="Insert the specified metrics to update into the .stats.json file even if they do not already exist.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help=(
            "Metric name to run. Can be passed multiple times to run only selected metrics. "
            "Defaults to all registered metrics."
        ),
    )
    parser.add_argument(
        "--acceptance-prob",
        type=float,
        default=None,
        help=(
            "Per-item inclusion probability for contextual gender bias subcollections. "
            "Enables sampling when provided (range: 0 to 1)."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Number of subcollections for contextual gender bias sampling (default: 30).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level to use (default: WARNING).",
    )
    args = parser.parse_args()
    if args.mode in {"language", "both"} and not args.language_key:
        parser.error("--language-key is required when --mode is language or both")
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info("Parsed arguments: %s", args)
    logger.info("Log level set to %s", args.log_level)

    # Handle input type
    if args.input_type == "huggingface":
        cache_dir = os.path.join("data", ".cache")
        logger.info("Input type is Hugging Face; using cache dir %s", cache_dir)
        args.input_file = fetch_huggingface_dataset(args.input_file, cache_dir)
        logger.info("Resolved Hugging Face input file to %s", args.input_file)

    # Compute output filename and handle existing data/errors
    base_path, _ = os.path.splitext(args.input_file)
    output_filename = f"{base_path}.stats.json"
    results_dict: Dict[str, Dict[str, Any]] = {}
    logger.info("Output will be written to %s", output_filename)

    if args.update:
        logger.info("Update mode enabled for metrics: %s", args.update)
        if not os.path.exists(output_filename):
            raise FileNotFoundError(
                f"Output file {output_filename} not found. "
                "The --update option requires an existing output file."
            )

        logger.info("Loading previous results from %s", output_filename)
        with open(output_filename, "r", encoding="utf-8") as f:
            results_dict = json.load(f)

        # Verify requested metrics exist in the current output
        for metric_to_update in args.update:
            if metric_to_update not in results_dict and not args.or_insert:
                raise KeyError(
                    f"Metric '{metric_to_update}' not found in existing results. "
                    f"Available metrics: {list(results_dict.keys())}"
                )
        logger.info("Existing results loaded successfully")

    # Determine key categorizations
    logger.info("Collecting all keys from input file")
    all_file_keys = get_all_keys_from_file(args.input_file)
    categorized_keys = set(args.text_keys + args.metadata_keys)
    logger.info("Found %d keys in input file", len(all_file_keys))

    languages = []
    if args.mode in {"language", "both"}:
        languages = get_all_languages_from_file(args.input_file, args.language_key)
        if not languages:
            raise ValueError(
                f"No non-empty language values found for key {args.language_key!r}."
            )
        logger.info("Found %d languages in input file", len(languages))

    key_types = {}
    for k in args.text_keys:
        key_types[k] = "text"
    for k in args.metadata_keys:
        key_types[k] = "metadata"
    for k in all_file_keys - categorized_keys:
        key_types[k] = "other"
    logger.info(
        "Key categorization complete: %d text, %d metadata, %d other",
        len(args.text_keys),
        len(args.metadata_keys),
        len(all_file_keys - categorized_keys),
    )

    # Register metrics
    logger.info("Registering metrics")
    registered_metrics = [
        HelloMetric(),
        LengthMetric(),
        AlphanumericMetric(),
        NumDocumentsMetric(),
        MetadataKeywordGroupsMetric(),
        VACARatioMetric(),
        TokenCountMetric(),
        VocabularySizeMetric(),
        StochasticDiversityMetric(),
        InnerOuterDiversityMetric(),
        MinhashDuplicationMetric(),
        MetadataYearCountMetric(),
        GenderBiasMetric(),
        ContextualGenderBiasMetric(
            acceptance_prob=args.acceptance_prob,
            sample_size=args.sample_size,
        ),
        STTRMetric(),
        MTLDMetric(),
    ]
    logger.info("Registered %d metrics", len(registered_metrics))

    if args.metric:
        logger.info("Metric filter requested: %s", args.metric)
        available_metrics = {metric.name: metric for metric in registered_metrics}
        invalid_metrics = [
            name for name in args.metric if name not in available_metrics
        ]
        if invalid_metrics:
            raise KeyError(
                f"Unknown metric name(s): {invalid_metrics}. "
                f"Available metrics: {list(available_metrics.keys())}"
            )

        selected_metric_names = list(dict.fromkeys(args.metric))
        registered_metrics = [available_metrics[name] for name in selected_metric_names]
        logger.info("Selected %d metrics after filtering", len(registered_metrics))

    # Map metrics to their target keys
    logger.info("Building task list")
    tasks = []
    for metric in registered_metrics:
        # Only schedule if updating specific metrics, otherwise schedule all
        if args.update and metric.name not in args.update:
            logger.info("Skipping metric '%s' due to update filter", metric.name)
            continue

        for key, ktype in key_types.items():
            if args.ignore_other and ktype == "other":
                logger.info(
                    "Skipping key '%s' for metric '%s' because --ignore-other is enabled",
                    key,
                    metric.name,
                )
                continue

            if ktype in metric.target_types or "all" in metric.target_types:
                if args.mode in {"global", "both"}:
                    tasks.append((metric, key, None))
                if args.mode in {"language", "both"}:
                    for language in languages:
                        tasks.append((metric.for_language(language), key, language))
                logger.debug("Scheduled metric '%s' for key '%s'", metric.name, key)

    if not tasks:
        logger.warning(
            "No tasks to process. Check your --metric/--update names or key configurations."
        )
        return

    logger.info("Scheduled %d tasks", len(tasks))

    # Using ThreadPoolExecutor for GPU compatibility
    logger.info("Starting task execution")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_metric_task,
                metric,
                args.input_file,
                key,
                args.language_key,
                language,
            ): (
                metric,
                key,
                language,
            )
            for metric, key, language in tasks
        }
        logger.info("Submitted %d futures", len(futures))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing tasks",
        ):
            metric, key, language = futures[future]
            metric_name = getattr(metric, "name", "unknown_metric")
            logger.info(
                "Processing completed task for metric '%s' and key '%s'",
                metric_name,
                key,
            )

            try:
                metric_name, key, language, result = future.result()
                logger.info(
                    "Task succeeded for metric '%s' and key '%s'", metric_name, key
                )
                store_result(
                    results_dict, args.mode, metric_name, key, language, result
                )
            except Exception as e:
                # On failure, record the error instead of breaking
                logger.exception(
                    "Task failed for metric '%s' and key '%s'", metric_name, key
                )
                key_for_error = key or "unknown_job"

                if args.develop:
                    # In develop mode, let exceptions propagate
                    raise RuntimeError(
                        f"Error processing metric '{metric_name}' for key '{key_for_error}': {e}"
                    ) from e

                store_result(
                    results_dict,
                    args.mode,
                    metric_name,
                    key_for_error,
                    language,
                    {"error": str(e)},
                )

            # Dump results to the output file after each job finishes
            logger.info("Writing intermediate results to %s", output_filename)
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2)

    logger.info("Analysis complete. Results written to %s", output_filename)


if __name__ == "__main__":
    main()
