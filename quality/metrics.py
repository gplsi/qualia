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


class BaseSemanticDiversityMetric(BaseMetric):
    """Shared semantic utilities for diversity-style metrics.

    Handles:
      - model loading
      - sentence splitting (spaCy if available, regex fallback)
      - cosine similarity helpers
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_seed: int = 42,
    ) -> None:
        import random
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._rng = random.Random(random_seed)
        self._sentence_splitter = None  # lazy init

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    # ---- Helpers shared by all semantic diversity-style metrics ----

    def _mean_pairwise_cosine(self, embeddings) -> float:
        """Compute mean pairwise cosine similarity for a 2D tensor/array."""
        import numpy as np

        emb = embeddings
        if isinstance(emb, list):
            emb = np.asarray(emb)

        n = emb.shape[0]
        if n < 2:
            return 1.0  # if only one vector, treat similarity as max

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb_norm = emb / norms

        sim = emb_norm @ emb_norm.T
        i_upper = np.triu_indices(n, k=1)
        pair_sims = sim[i_upper]
        return float(pair_sims.mean()) if pair_sims.size > 0 else 1.0

    def _cosine_two(self, v1, v2) -> float:
        """Cosine similarity between two 1D vectors."""
        import numpy as np

        num = float(v1 @ v2)
        den = float((np.linalg.norm(v1) + 1e-8) * (np.linalg.norm(v2) + 1e-8))
        return num / den

    def _get_sentence_splitter(self):
        """Return a callable(text) -> List[str] that splits into sentences."""
        if self._sentence_splitter is not None:
            return self._sentence_splitter

        import re

        pattern = re.compile(r"[.!?]+\s+")

        def regex_split(text: str):
            parts = pattern.split(text)
            return [p.strip() for p in parts if p.strip()]

        self._sentence_splitter = regex_split
        return self._sentence_splitter


class StochasticDiversityMetric(BaseSemanticDiversityMetric):
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
        if granularity not in ("text", "sentence"):
            raise ValueError("granularity must be 'text' or 'sentence'")
        super().__init__(model_name=model_name, random_seed=random_seed)
        self._sample_prob = sample_prob
        self._max_pool_size = max_pool_size
        self._eval_interval = eval_interval
        self._granularity = granularity

    @property
    def name(self) -> str:
        return f"{self._granularity}-semantic-diversity"

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

            if self._granularity == "text":
                units = [text]
            else:  # "sentence"
                units = splitter(text)

            for unit in units:
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
            return 0.0

        s = float(np.mean(snapshots))
        diversity = 1.0 - s
        return max(0.0, min(1.0, diversity))


class InnerOuterDiversityMetric(BaseSemanticDiversityMetric):
    """Computes:
    - inner diversity: 1 - mean(sim(sent_i, sent_j)) over all sentence
      pairs *within* each value, averaged across values.
    - outer diversity: 1 - mean(sim(sent_a, sent_b)) over sentence pairs
      drawn from *different* values.

    This implementation is streaming and bounded in memory:

      - For each value we:
          * split into sentences
          * randomly keep up to `sentences_per_value` sentences
          * encode them once and compute that value's inner diversity
      - A fixed-size pool of values is maintained. Each pooled value stores
        its encoded sentence embeddings only.
      - Values are added to the pool with probability `value_accept_prob`.
        When the pool is full, a random value is evicted.
      - Whenever a value is added to the pool, we compute outer similarities
        between that new value and a few randomly chosen existing values,
        using up to `sentence_pairs_per_value_pair` random sentence pairs
        per (new, existing) pair.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_seed: int = 42,
        inner_sentences_per_value: int | None = None,  # None => use all sentences
        outer_sentences_per_value: int = 8,
        pool_max_values: int = 64,
        value_accept_prob: float = 0.5,
        sentence_pairs_per_value_pair: int = 4,
    ) -> None:
        super().__init__(model_name=model_name, random_seed=random_seed)
        if (
            inner_sentences_per_value is not None
            and outer_sentences_per_value > inner_sentences_per_value
        ):
            raise ValueError(
                "outer_sentences_per_value must be <= inner_sentences_per_value "
                "when inner_sentences_per_value is not None"
            )
        self._inner_sentences_per_value = inner_sentences_per_value
        self._outer_sentences_per_value = outer_sentences_per_value
        self._pool_max_values = pool_max_values
        self._value_accept_prob = value_accept_prob
        self._sentence_pairs_per_value_pair = sentence_pairs_per_value_pair

    @property
    def name(self) -> str:
        return "inner-outer-diversity"

    def compute(self, values: Iterable[Any]) -> Any:
        import numpy as np

        rng = self._rng
        splitter = self._get_sentence_splitter()

        inner_sims: List[float] = []
        pool: List[dict] = []
        outer_sims: List[float] = []

        for v in values:
            if v is None:
                continue

            text = str(v).strip()
            if not text:
                continue

            sentences = splitter(text)
            if not sentences:
                continue

            # --------- Inner diversity sentences (possibly all) ----------
            inner_sentences = sentences
            if (
                self._inner_sentences_per_value is not None
                and len(sentences) > self._inner_sentences_per_value
            ):
                inner_sentences = rng.sample(sentences, self._inner_sentences_per_value)

            # Encode inner sentences once
            inner_emb = self._model.encode(
                inner_sentences,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            if inner_emb.shape[0] >= 2:
                s_val = self._mean_pairwise_cosine(inner_emb)
                inner_sims.append(s_val)

            # --------- Pool / outer diversity control ----------
            if rng.random() > self._value_accept_prob:
                continue

            # Derive outer_emb by subsampling rows from inner_emb
            if inner_emb.shape[0] == 0:
                continue

            if inner_emb.shape[0] > self._outer_sentences_per_value:
                # sample indices from the existing inner_emb rows
                idxs = rng.sample(
                    range(inner_emb.shape[0]), self._outer_sentences_per_value
                )
                outer_emb = inner_emb[idxs]
            else:
                outer_emb = inner_emb

            if outer_emb.shape[0] == 0:
                continue

            # Evict if pool full
            if len(pool) >= self._pool_max_values:
                drop_idx = rng.randrange(len(pool))
                pool.pop(drop_idx)

            # Compute outer similarities with existing pool before adding
            if pool:
                max_partners = min(len(pool), self._pool_max_values)
                n_partners = min(len(pool), max_partners)
                partner_indices = rng.sample(range(len(pool)), n_partners)

                for idx in partner_indices:
                    other_emb = pool[idx]["emb"]
                    if other_emb.shape[0] == 0 or outer_emb.shape[0] == 0:
                        continue

                    for _ in range(self._sentence_pairs_per_value_pair):
                        i = rng.randrange(outer_emb.shape[0])
                        j = rng.randrange(other_emb.shape[0])
                        v1 = outer_emb[i]
                        v2 = other_emb[j]
                        outer_sims.append(self._cosine_two(v1, v2))

            # Finally insert into pool
            pool.append({"emb": outer_emb})

        # --- Aggregate inner and outer similarities into diversity scores ---
        if inner_sims:
            inner_mean_sim = float(np.mean(inner_sims))
            inner_div = max(0.0, min(1.0, 1.0 - inner_mean_sim))
        else:
            inner_div = 0.0

        if outer_sims:
            outer_mean_sim = float(np.mean(outer_sims))
            outer_div = max(0.0, min(1.0, 1.0 - outer_mean_sim))
        else:
            outer_div = 0.0

        return {"inner": inner_div, "outer": outer_div}
