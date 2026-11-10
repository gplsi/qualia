import re
from pathlib import Path
from typing import Iterable, Any, List, Set, Dict

from bfair.metrics.lm import (
    BiasScore,
    EnglishGenderedWords,
    FixedContext,
    InfiniteContext,
    ContinualContext,
    SpanishGenderedWords,
)

from quality.metrics.base import BaseMetric


class GenderBiasMetric(BaseMetric):
    """
    Computes gender bias by counting occurrences of male and female-related words
    and returning their counts along with the female-to-male ratio.
    """

    def __init__(self, base_dir: str = "data/gender_bias", sub_dirs: List[str] = None):
        """
        Initializes the metric and pre-loads the vocabularies.

        Args:
            base_dir: The root directory for the gender bias data.
            sub_dirs: List of subdirectories to check for 'female.csv' and 'male.csv'.
        """
        self.base_dir = Path(base_dir)
        self.sub_dirs = (
            sub_dirs
            if sub_dirs is not None
            else [dir.name for dir in self.base_dir.iterdir() if dir.is_dir()]
        )

        # Pre-compile regex for fast tokenization
        self._tokenizer = re.compile(r"\b\w+\b")

        # Load vocabularies into sets for O(1) lookups
        # Use uppercase for case-sensitive and lowercase for case-insensitive vocabularies
        self.ci_female_words = self._load_vocabulary("female.csv")
        self.cs_female_words = self._load_vocabulary("FEMALE.csv")
        self.ci_male_words = self._load_vocabulary("male.csv")
        self.cs_male_words = self._load_vocabulary("MALE.csv")

    def _load_vocabulary(self, filename: str) -> Set[str]:
        """Reads CSV files across the configured subdirectories and builds a set of words."""
        vocab = set()
        for sub_dir in self.sub_dirs:
            file_path = self.base_dir / sub_dir / filename
            if file_path.is_file():
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read line by line, clean whitespace, and convert to lowercase
                    for line in f:
                        word = line.strip().lower()
                        if word:  # Skip empty lines if there are any
                            vocab.add(word)
        return vocab

    @property
    def name(self) -> str:
        return "gender-bias"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Dict[str, Any]:
        female_count = 0
        male_count = 0

        for val in values:
            if not isinstance(val, str):
                continue

            tokens = self._tokenizer.findall(val)
            for token in tokens:
                if (
                    token.lower() in self.ci_female_words
                    or token in self.cs_female_words
                ):
                    female_count += 1
                if token.lower() in self.ci_male_words or token in self.cs_male_words:
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


class ContextualGenderBiasMetric(BaseMetric):
    """Compute contextual count-disparity and log-score gender bias metrics."""

    CONTEXTS = {
        "fixed": FixedContext,
        "infinite": InfiniteContext,
        "continual": ContinualContext,
        "continuous": ContinualContext,
    }
    GROUP_WORDS = {
        "english": EnglishGenderedWords,
        "spanish": SpanishGenderedWords,
    }

    def __init__(
        self,
        language: str = "english",
        context: Any = "continuous",
        group_words: Any = None,
        use_root: bool = True,
    ):
        self.language = language.lower()
        self.context = context
        self.group_words = group_words
        self.use_root = use_root
        self._bias_score = None

    @property
    def name(self) -> str:
        return "contextual-gender-bias"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _get_bias_score(self) -> BiasScore:
        if self._bias_score is None:
            group_words = self.group_words
            if group_words is None:
                try:
                    group_words = self.GROUP_WORDS[self.language]().get_group_words()
                except KeyError as error:
                    raise ValueError(
                        f"No default gender seed words for {self.language!r}; "
                        "provide group_words explicitly."
                    ) from error

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
