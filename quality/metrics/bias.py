import re
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
    ):
        super().__init__(language=language, group_words=group_words)
        self.context = context
        self.use_root = use_root
        self._bias_score = None

    @property
    def name(self) -> str:
        return "contextual-gender-bias"

    def _language_options(self) -> Dict[str, Any]:
        return {"context": self.context, "use_root": self.use_root}

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

    def compute(self, values: Iterable[Any]) -> Dict[str, Dict[str, float]]:
        self._require_language()
        texts = [value for value in values if isinstance(value, str)]
        if not texts:
            return {
                mode: {"mean": 0.0, "stdev": 0.0}
                for mode in (BiasScore.S_RATIO, BiasScore.S_LOG)
            }

        bias_score = self._get_bias_score()
        bias_score.group_words.reset_context()
        scores, _ = bias_score(texts)
        return {
            mode: {"mean": float(scores[mode][0]), "stdev": float(scores[mode][1])}
            for mode in (BiasScore.S_RATIO, BiasScore.S_LOG)
        }
