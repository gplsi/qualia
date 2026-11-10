# Qualia - Quality for Language & AI Toolkit

Qualia is a small, extensible toolkit to compute dataset quality metrics over JSONL text corpora or Hugging Face datasets. It streams over your data and writes aggregated statistics (length, token counts, semantic diversity, duplication, metadata coverage, gender bias, etc.) into a single `<input>.stats.json` file.

## Why this project is useful

- **Quick quality pass** on new or existing text datasets without writing custom scripts.
- **Extensible metric framework**: implement a new metric by subclassing a simple `BaseMetric` and registering it.
- **Streaming and scalable**: processes JSONL files lazily instead of loading everything into memory.
- **Works with local files and Hugging Face datasets** via the same CLI.
- **Rich built‑in metrics** covering basic text stats, metadata, duplication, semantic diversity, dialect ratios, and gender bias.

## Installation

This project is a plain Python package (no installer yet). The easiest way to get started is to clone the repository and install dependencies into a virtual environment.

```bash
# Clone the repository
git clone <THIS_REPO_URL>
cd quality

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install datasets tqdm mmh3 sentence-transformers numpy tiktoken

# Install disvaca from the vendored copy (once present)
# This repository expects a local checkout of disvaca under vendor/disvaca
pip install -e vendor/disvaca
```

Depending on your environment, `sentence-transformers` may additionally pull in `torch` or `tensorflow`. For GPU acceleration, ensure that the appropriate CUDA-enabled PyTorch build is installed.

## Basic usage

The main entry point is the `quality` module, which can be executed directly with `python -m quality`. It expects a JSONL file where each line is a JSON object and writes a corresponding `.stats.json` file next to it.

```bash
python -m quality data/example.jsonl \
  --text-keys text1 \
  --text-keys text2 \
  --text-keys text3 \
  --text-keys text4 \
  --metadata-keys metadata
```

This command will:

- Read `data/example.jsonl` line by line.
- Treat the fields `text1`, `text2`, `text3`, `text4` as text columns.
- Treat `metadata` as a metadata column.
- Run all registered metrics over the appropriate columns.
- Write results to `data/example.stats.json`.

By default, metrics are computed over the global corpus and no language key is needed.
Use `--mode language` to compute separate results for each language, or `--mode both` to
include global and per-language results. Both modes require `--language-key KEY` to identify the language field in each record.
In all modes the result shape is `metric -> field -> language`.

```bash
python -m quality data/example.jsonl \
  --text-keys text \
  --language-key language \
  --mode both
```

You can inspect `data/example.jsonl` and `data/example.stats.json` in the repository for a concrete input/output example.

### Using Hugging Face datasets

Instead of a local JSONL file, you can also pass a Hugging Face dataset name and ask the tool to fetch and cache it as JSONL automatically:

```bash
python -m quality your-dataset-name \
  --input-type huggingface \
  --text-keys text \
  --metadata-keys metadata
```

This will:

- Download the `train` split using `datasets.load_dataset`.
- Cache it under `data/.cache/your-dataset-name/data.jsonl`.
- Run metrics on the cached JSONL file.

### Selecting column types

`quality` needs to know which keys in your JSONL records are text vs. metadata. Use:

- `--text-keys KEY` (can be repeated).
- `--metadata-keys KEY` (can be repeated).

Any remaining keys that appear in the first non-empty line of the JSONL file are automatically categorized as `other` and are only processed by metrics that support `other` or `all` as a target type.

### Updating a subset of metrics

If you already have a `.stats.json` file and want to recompute only some metrics (for example, after adding a new metric or fixing a bug), you can use `--update` with metric names:

```bash
python -m quality data/example.jsonl \
  --text-keys text1 --text-keys text2 \
  --metadata-keys metadata \
  --update token-count --update gender-bias
```

- The tool will read `data/example.stats.json`.
- Only `token-count` and `gender-bias` will be recomputed and updated in the existing results.
- If a requested metric is not present in the file, the command will fail with a clear error.

### Develop mode

By default, errors in a particular metric/key combination are recorded in the output file under an `"error"` field and processing continues. For interactive debugging, you can enable develop mode:

```bash
python -m quality data/example.jsonl --text-keys text1 --develop
```

With `--develop`, metric exceptions are raised immediately so you can see full stack traces.

## Available metrics

The following metrics are registered by default (see `quality/metrics` for implementations):

- **HelloMetric** (`hello`): toy metric that demonstrates the interface.
- **LengthMetric** (`length`): total character length of text fields.
- **AlphanumericMetric** (`alphanumeric`): counts alphanumeric characters.
- **MetadataKeywordGroupsMetric** (`metadata-keyword-groups`): checks whether metadata fields contain typical keywords (date, source, author, language, license).
- **VACARatioMetric** (`vaca-ratio`): uses `disvaca` to compute Valencian/Catalan dialect ratios (micro and macro averages).
- **TokenCountMetric** (`token-count`): counts approximate LLM-style tokens using `tiktoken`.
- **VocabularySizeMetric** (`vocabulary-size`): number of distinct whitespace-delimited tokens.
- **StochasticDiversityMetric** (`text-semantic-diversity` / sentence variants): estimates semantic diversity as `1 - mean cosine similarity` over embeddings from a SentenceTransformers model.
- **InnerOuterDiversityMetric** (`inner-outer-diversity`): computes inner (within-document) and outer (across-documents) semantic diversity.
- **MinhashDuplicationMetric** (`minhash-duplication`): approximate global duplication score using a MinHash-style sketch over character shingles.
- **MetadataYearCountMetric** (`metadata-year-count`): extracts and counts years (and ranges, e.g. `2010-2012`) from metadata, producing a `{year: count}` mapping.
- **GenderBiasMetric** (`gender-bias`): counts male- and female-associated terms using vocabularies in `data/gender_bias`, returning counts, percentages, and a ratio.

All metrics declare the column types they support via `target_types` (e.g. `"text"`, `"metadata"`). The dispatcher only applies each metric to compatible columns.

## Running on a Slurm cluster

The `slurm` directory contains helper scripts to run quality checks as batch jobs.

A simple example is `slurm/example.sh`:

```bash
#!/bin/bash

JOB_NAME="quality-example"
COMMAND="python -m quality data/example.jsonl  --text-keys text1 --text-keys text2 --text-keys text3 --text-keys text4 --metadata-keys metadata"

sbatch \
  --job-name="$JOB_NAME" \
  --export=ALL,COMMAND="$COMMAND" \
  ./slurm/job.sh
```

You can adapt this pattern to your own cluster settings:

1. Edit `JOB_NAME` and `COMMAND` in one of the Slurm scripts or create your own.
2. Submit with `sbatch` from the project root.
3. Monitor the job as usual on your HPC system.

## Extending the toolkit with new metrics

To add a new metric:

1. Create a new module under `quality/metrics/` (or extend an existing one).
2. Subclass `BaseMetric` and implement:
   - `name` (used as the top-level key in the `.stats.json` output).
   - `target_types` (list of supported column types, e.g. `["text"]`).
   - `compute(values: Iterable[Any])` which consumes a lazy stream of values and returns any JSON-serializable result.
3. Import and expose the metric in `quality/metrics/__init__.py`.
4. Instantiate your metric in the `registered_metrics` list in `quality/analyze.py`.

Your metric will then automatically participate in the CLI, including `--update`.

## Getting help

- Browse the source code under `quality/` and the examples under `data/` for reference.
- If this repository is hosted on GitHub or a similar forge, consider opening an issue with:
  - a short description of the problem,
  - the command you ran,
  - a minimal example of your input format.

## Maintainers and contributions

- **Maintainer**: Juan Pablo Consuegra-Ayala (jpconsuegra93@gmail.com).
- **Contributions**: Contributions are welcome. Typical ways to contribute include:
  - Adding new metrics for other quality aspects.
  - Improving documentation and examples.
  - Extending support for additional input formats or metadata conventions.

## Funding

The development of this framework has been partially funded by the *Ministerio para la Transformación Digital y de la Función Pública* — Funded by **EU – NextGenerationEU**, within the framework of the project *Desarrollo de Modelos ALIA*.

## Acknowledgments

This toolkit was developed as part of research conducted at the **GPLSI** (Grupo de Procesamiento del Lenguaje y Sistemas de Información) at the **University of Alicante**.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) files for more details.