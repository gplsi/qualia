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


class VocabularySizeMetric(BaseMetric):
    """Counts the number of distinct whitespace-delimited tokens in text fields."""

    @property
    def name(self) -> str:
        return "vocabulary-size"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        vocab = set()
        for v in values:
            if v is None:
                continue
            vocab.update(str(v).split())
        return len(vocab)


class SemanticDiversityMetric(BaseMetric):
    """Estimates semantic diversity as 1 - s, where s is the mean of
    pairwise cosine similarities between sentence embeddings.

    It uses:
      - random subsampling of sentences/text units,
      - a bounded pool size,
      - periodic evaluation of mean similarity on the pool.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sample_prob: float = 0.25,
        max_pool_size: int = 256,
        eval_interval: int = 32,
        random_seed: int = 42,
        granularity: str = "text",  # "text" or "sentence"
    ) -> None:
        import random
        from sentence_transformers import SentenceTransformer

        if granularity not in ("text", "sentence"):
            raise ValueError("granularity must be 'text' or 'sentence'")

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._sample_prob = sample_prob
        self._max_pool_size = max_pool_size
        self._eval_interval = eval_interval
        self._rng = random.Random(random_seed)
        self._granularity = granularity
        self._sentence_splitter = None  # lazy init

    @property
    def name(self) -> str:
        return "semantic-diversity"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _mean_pairwise_cosine(self, embeddings) -> float:
        """Compute mean pairwise cosine similarity for a 2D tensor/array."""
        import numpy as np

        emb = embeddings
        if isinstance(emb, list):
            emb = np.asarray(emb)

        n = emb.shape[0]
        if n < 2:
            return 1.0  # if only one sentence, treat similarity as max

        # normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_norm = emb / norms

        # cosine similarity matrix
        sim = emb_norm @ emb_norm.T
        # take upper triangle without diagonal
        i_upper = np.triu_indices(n, k=1)
        pair_sims = sim[i_upper]
        return float(pair_sims.mean()) if pair_sims.size > 0 else 1.0

    def _get_sentence_splitter(self):
        """Return a callable(text) -> List[str] that splits into sentences.

        Tries spaCy if available, otherwise falls back to a regex-based splitter.
        """
        if self._sentence_splitter is not None:
            return self._sentence_splitter

        # Try spaCy (fast Cython seg + rule-based), but keep it optional
        try:
            import spacy  # type: ignore

            try:
                nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
            except OSError:
                # model not downloaded; fall back to blank
                nlp = spacy.blank("en")
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")

            def spacy_split(text: str):
                doc = nlp(text)
                return [s.text.strip() for s in doc.sents if s.text.strip()]

            self._sentence_splitter = spacy_split
            return self._sentence_splitter
        except Exception:
            # Fallback: smarter regex splitter
            import re

            # Very lightweight heuristic:
            #   - Do not split after common abbreviations
            #   - Split on [.!?] followed by whitespace + capital letter or quote
            abbrevs = r"(?:Mr|Ms|Mrs|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)"
            pattern = re.compile(
                rf"""
                (?<!{abbrevs})      # negative lookbehind for abbreviations
                (?<=[.!?])         # sentence end punctuation
                ["')\]]*           # optional closing quotes/brackets
                \s+                # whitespace
                (?=[A-Z0-9"])      # next sentence starts with capital/number/quote
                """,
                re.VERBOSE,
            )

            def regex_split(text: str):
                parts = pattern.split(text)
                return [p.strip() for p in parts if p.strip()]

            self._sentence_splitter = regex_split
            return self._sentence_splitter

    def compute(self, values: Iterable[Any]) -> Any:
        import numpy as np

        pool: List[str] = []
        snapshots: List[float] = []
        accepted_since_eval = 0

        splitter = self._get_sentence_splitter()

        for v in values:
            if v is None:
                continue

            text = str(v).strip()
            if not text:
                continue

            # choose units according to granularity
            if self._granularity == "text":
                units = [text]
            else:  # "sentence"
                units = splitter(text)

            for unit in units:
                # random subsampling at unit level
                if self._rng.random() > self._sample_prob:
                    continue

                if len(pool) < self._max_pool_size:
                    pool.append(unit)
                else:
                    idx = self._rng.randrange(len(pool))
                    pool[idx] = unit

                accepted_since_eval += 1

                if accepted_since_eval >= self._eval_interval and len(pool) >= 2:
                    embeddings = self._model.encode(
                        pool,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    snapshots.append(self._mean_pairwise_cosine(embeddings))
                    accepted_since_eval = 0

        if snapshots == [] and len(pool) >= 2:
            embeddings = self._model.encode(
                pool,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            snapshots.append(self._mean_pairwise_cosine(embeddings))

        if not snapshots:
            # not enough data to estimate diversity; return 0.0
            return 0.0

        s = float(np.mean(snapshots))
        diversity = 1.0 - s
        # clamp for numerical safety
        return max(0.0, min(1.0, diversity))
