import re
import random
from itertools import chain
from numbers import Real
from statistics import mean, median, pstdev
from typing import Iterable, Any, List, Dict

from bfair.metrics.lm import (
    BiasScore,
    EnglishGenderedWords,
    FixedContext,
    InfiniteContext,
    ContinualContext,
    SpanishGenderedWords,
    ValencianGenderedWords,
)

from quality.metrics.base import BaseMetric


class BaseGenderBiasMetric(BaseMetric):
    GROUP_WORDS = {
        "english": EnglishGenderedWords,
        "spanish": SpanishGenderedWords,
        "valencian": ValencianGenderedWords,
        "catalan": ValencianGenderedWords,
    }
    LANGUAGE_ALIASES = {
        "en": "english",
        "eng": "english",
        "es": "spanish",
        "spa": "spanish",
        "ca": "catalan",
        "cat": "catalan",
        "va": "valencian",
        "val": "valencian",
    }

    def __init__(self, language: str = None, group_words: Any = None):
        self.language = language.lower() if language else language
        self.group_words = group_words
        self._tokenizer = re.compile(r"\b\w+\b")
        self._group_words_cache = None
        self._male_words = None
        self._female_words = None

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _language_options(self) -> Dict[str, Any]:
        return {}

    def for_language(self, language: str) -> "BaseGenderBiasMetric":
        language_code = language.strip().lower().replace("_", "-").split("-", 1)[0]
        configured_language = self.LANGUAGE_ALIASES.get(language_code, language_code)
        return type(self)(
            language=configured_language,
            group_words=self.group_words,
            **self._language_options(),
        )

    def _get_group_words(self):
        if self._group_words_cache is not None:
            return self._group_words_cache
        if self.group_words is None:
            try:
                self._group_words_cache = self.GROUP_WORDS[
                    self.language
                ]().get_group_words()
            except KeyError as error:
                raise ValueError(
                    f"No default gender seed words for {self.language!r}; "
                    "provide group_words explicitly."
                ) from error
        else:
            self._group_words_cache = self.group_words
        return self._group_words_cache

    def _get_gender_words(self):
        if self._female_words is None:
            group_words = self._get_group_words()
            self._female_words = self._normalize_vocabulary(
                group_words.get_words("female")
            )
            self._male_words = self._normalize_vocabulary(
                group_words.get_words("male")
            )
        return self._female_words, self._male_words

    @staticmethod
    def _normalize_vocabulary(words):
        return {
            word.strip().lower()
            for word in words
            if isinstance(word, str) and word.strip()
        }

    def _require_language(self) -> None:
        if not self.language:
            raise ValueError(
                f"Language must be specified for {type(self).__name__}. "
                "Use the 'for_language' method to configure the metric."
            )


class GenderBiasMetric(BaseGenderBiasMetric):
    """Count gendered words and report their proportions and balance ratio."""

    @property
    def name(self) -> str:
        return "gender-bias"

    def compute(self, values: Iterable[Any]) -> Dict[str, Any]:
        self._require_language()
        female_words, male_words = self._get_gender_words()
        female_count = 0
        male_count = 0

        for val in values:
            if not isinstance(val, str):
                continue

            tokens = self._tokenizer.findall(val)
            for token in tokens:
                if token.lower() in female_words:
                    female_count += 1
                if token.lower() in male_words:
                    male_count += 1

        if female_count == 0 and male_count == 0:
            female_percent = male_percent = 0
            ratio = 1.0
        else:
            female_percent = female_count / (female_count + male_count) * 100
            male_percent = male_count / (female_count + male_count) * 100
            if female_count == 0 or male_count == 0:
                ratio = "inf"
            else:
                ratio = min(female_count, male_count) / max(female_count, male_count)

        return {
            "female-count": female_count,
            "male-count": male_count,
            "ratio": ratio,
            "female-percent": female_percent,
            "male-percent": male_percent,
        }


class ContextualGenderBiasMetric(BaseGenderBiasMetric):
    """Compute contextual count-disparity and log-score gender bias metrics."""

    CONTEXTS = {
        "fixed": FixedContext,
        "infinite": InfiniteContext,
        "continual": ContinualContext,
        "continuous": ContinualContext,
    }

    def __init__(
        self,
        language: str = None,
        context: Any = "continuous",
        group_words: Any = None,
        use_root: bool = True,
        acceptance_prob: float = None,
        sample_size: int = 30,
        seed: int = None,
    ):
        if acceptance_prob is not None and (
            isinstance(acceptance_prob, bool)
            or not isinstance(acceptance_prob, Real)
            or not 0 <= acceptance_prob <= 1
        ):
            raise ValueError("acceptance_prob must be a number between 0 and 1")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or None")
        super().__init__(language=language, group_words=group_words)
        self.context = context
        self.use_root = use_root
        self.acceptance_prob = acceptance_prob
        self.sample_size = sample_size
        self.seed = seed
        self._bias_score = None

    @property
    def name(self) -> str:
        return "contextual-gender-bias"

    def _language_options(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "use_root": self.use_root,
            "acceptance_prob": self.acceptance_prob,
            "sample_size": self.sample_size,
            "seed": self.seed,
        }

    def _get_bias_score(self) -> BiasScore:
        if self._bias_score is None:
            group_words = self._get_group_words()
            context = self.context
            if isinstance(context, str):
                try:
                    context = self.CONTEXTS[context.lower()]()
                except KeyError as error:
                    raise ValueError(
                        f"Unsupported context {context!r}; "
                        f"choose from {', '.join(self.CONTEXTS)}."
                    ) from error

            self._bias_score = BiasScore(
                language=self.language,
                group_words=group_words,
                context=context,
                scoring_modes=[BiasScore.S_RATIO, BiasScore.S_LOG],
                use_root=self.use_root,
                semantic_check=False,
            )
        return self._bias_score

    def compute(self, values: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        self._require_language()
        if self.acceptance_prob is None:
            return self._compute_sample(
                value for value in values if isinstance(value, str)
            )

        rng = random.Random(self.seed)
        subcollections = [[] for _ in range(self.sample_size)]
        for value in values:
            if isinstance(value, str):
                for subcollection in subcollections:
                    if rng.random() < self.acceptance_prob:
                        subcollection.append(value)

        results = [self._compute_sample(iter(sample)) for sample in subcollections]
        report = {
            mode: {
                key: {
                    "min": min(subcollection_values),
                    "max": max(subcollection_values),
                    "mean": mean(subcollection_values),
                    "stdev": pstdev(subcollection_values),
                    "median": median(subcollection_values),
                }
                for key in ("mean", "stdev")
                for subcollection_values in [
                    [result[mode][key] for result in results]
                ]
            }
            for mode in (BiasScore.S_RATIO, BiasScore.S_LOG)
        }
        subcollection_lengths = [len(sample) for sample in subcollections]
        report["support"] = {
            "sample_size": self.sample_size,
            "min": min(subcollection_lengths),
            "max": max(subcollection_lengths),
            "mean": mean(subcollection_lengths),
            "stdev": pstdev(subcollection_lengths),
            "median": median(subcollection_lengths),
        }
        return report

    def _compute_sample(
        self, texts: Iterable[str]
    ) -> Dict[str, Dict[str, float]]:
        first_text = next(iter(texts), None)
        if first_text is None:
            return {
                mode: {"mean": 0.0, "stdev": 0.0}
                for mode in (BiasScore.S_RATIO, BiasScore.S_LOG)
            }

        bias_score = self._get_bias_score()
        bias_score.group_words.reset_context()
        scores, _ = bias_score(chain((first_text,), texts))
        return {
            mode: {"mean": float(scores[mode][0]), "stdev": float(scores[mode][1])}
            for mode in (BiasScore.S_RATIO, BiasScore.S_LOG)
        }
