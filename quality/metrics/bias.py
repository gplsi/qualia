import re
from pathlib import Path
from typing import Iterable, Any, List, Set, Dict

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
        self.female_words = self._load_vocabulary("female.csv")
        self.male_words = self._load_vocabulary("male.csv")

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
            else:
                print(f"Warning: Word list not found at {file_path}")
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

            tokens = self._tokenizer.findall(val.lower())
            for token in tokens:
                if token in self.female_words:
                    female_count += 1
                if token in self.male_words:
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
