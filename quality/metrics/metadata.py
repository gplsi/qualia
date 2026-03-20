
import re
from collections import Counter
from typing import Iterable, Any, List

from quality.metrics.base import BaseMetric


class MetadataKeywordGroupsMetric(BaseMetric):
    """Generic metric for checking groups of metadata keywords.

    For each configured group, it returns True iff:
      - at least one value was seen, and
      - every non-empty value contained at least one of that group's keywords.

    Result format:
      {
        "<group_name>": bool,
        ...
      }
    """

    def __init__(self, groups: dict[str, List[str]] | None = None) -> None:
        """If `groups` is None, use the default metadata keyword groups."""
        if groups is None:
            groups = self.default_groups()
        # normalize keywords to lowercase
        self._groups: dict[str, List[str]] = {
            gname: [k.lower() for k in keywords] for gname, keywords in groups.items()
        }

    @staticmethod
    def default_groups() -> dict[str, List[str]]:
        """Return a default set of metadata keyword groups.

        These are aimed at common fields like dates, source, author, etc.
        """
        return {
            "date": [
                "date",
                "fecha",
                "jahr",
                "année",
                "data",
                "published",
                "publication",
                "created",
                "modified",
                "updated",
                "timestamp",
                "time",
                "year",
            ],
            "source": [
                "source",
                "origen",
                "quelle",
                "origine",
                "provenance",
                "provider",
                "publisher",
                "journal",
                "magazine",
                "newspaper",
                "website",
                "url",
                "link",
                "doi",
            ],
            "author": [
                "author",
                "autores",
                "creator",
                "created by",
                "byline",
                "editor",
                "contributor",
                "writer",
                "researcher",
            ],
            "language": [
                "language",
                "lang",
                "idioma",
                "sprache",
                "langue",
                "locale",
                "iso639",
            ],
            "license": [
                "license",
                "licencia",
                "lizenz",
                "licence",
                "copyright",
                "usage",
                "rights",
                "cc-by",
                "cc0",
                "public domain",
                "permission",
            ],
        }

    @property
    def name(self) -> str:
        # Generic base; concrete subclasses should override
        return "metadata-keyword-groups"

    @property
    def target_types(self) -> List[str]:
        return ["metadata"]

    def compute(self, values: Iterable[Any]) -> Any:
        # Track:
        #   - whether we saw any non-empty value at all
        #   - for each group, whether ALL seen values matched that group
        seen_any = False
        group_all_true: dict[str, bool] = {g: True for g in self._groups}

        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue

            seen_any = True
            lower_text = text.lower()

            for gname, keywords in self._groups.items():
                if not group_all_true[gname]:
                    # already known to be False, no need to re-check
                    continue
                if not any(kw in lower_text for kw in keywords):
                    group_all_true[gname] = False

        if not seen_any:
            # No data: all groups are False
            return {g: False for g in self._groups}

        return group_all_true


class MetadataYearCountMetric(BaseMetric):
    """Counts how many metadata values fall into each year.

    For each metadata value:
      - If it is a dict and contains a date-like key, use that value.
      - Otherwise, treat the whole metadata value as text.
    From the chosen text, we:
      - detect explicit 4-digit years within a reasonable range,
      - detect year ranges like '1990-1993' or '1990–1993'
        (must be explicitly 4 digits on both sides) and count all years in the range.
    The result is a mapping {year: count}.
    """

    def __init__(
        self,
        min_year: int = 1800,
        max_year: int = 2100,
    ) -> None:
        self._min_year = min_year
        self._max_year = max_year
        self._Counter = Counter

        # Regex for single 4-digit years
        self._year_re = re.compile(r"\b(18|19|20)\d{2}\b")

        # Regex for ranges: 1990-1993, 1990–1993, etc.
        # Captures: century, start_suffix, 4-digit end_year
        self._range_re = re.compile(
            r"\b(18|19|20)(\d{2})\s*[-/\u2013\u2014]\s*(\d{4})\b"
        )

        # Common date-like keys in metadata dicts
        self._date_keys = {
            "date",
            "fecha",
            "year",
            "jahr",
            "année",
            "data",
            "published",
            "publication",
            "created",
            "modified",
            "updated",
            "timestamp",
            "time",
        }

    @property
    def name(self) -> str:
        return "metadata-year-count"

    @property
    def target_types(self) -> List[str]:
        return ["metadata"]

    # ---------------- internal helpers ----------------

    def _extract_text_from_metadata(self, meta: Any) -> str | None:
        """Return the most date-like textual representation from metadata."""
        if isinstance(meta, dict):
            for key in meta:
                if str(key).lower() in self._date_keys:
                    val = meta[key]
                    if val is not None:
                        txt = str(val).strip()
                        if txt:
                            return txt

            parts = []
            for v in meta.values():
                if v is not None:
                    s = str(v).strip()
                    if s:
                        parts.append(s)
            return " ".join(parts) if parts else None

        if meta is None:
            return None
        txt = str(meta).strip()
        return txt or None

    def _extract_years_from_text(self, text: str) -> list[int]:
        """Extract all plausible years from a textual field."""
        years: list[int] = []

        # 1) Ranges first (requires 4 digits on both sides)
        for m in self._range_re.finditer(text):
            century = m.group(1)  # e.g., "19"
            start_suffix = m.group(2)  # e.g., "90"
            end_part = m.group(3)  # e.g., "1993"

            start_year = int(century + start_suffix)
            end_year = int(end_part)

            # If they are typed backward (e.g., 1993-1990), swap them
            if start_year > end_year:
                start_year, end_year = end_year, start_year

            for y in range(start_year, end_year + 1):
                if self._min_year <= y <= self._max_year:
                    years.append(y)

        # 2) Standalone years (including those already in ranges; we'll dedup later)
        for m in self._year_re.finditer(text):
            y = int(m.group(0))
            if self._min_year <= y <= self._max_year:
                years.append(y)

        # Deduplicate while preserving order
        seen = set()
        uniq_years: list[int] = []
        for y in years:
            if y not in seen:
                seen.add(y)
                uniq_years.append(y)
        return uniq_years

    # ---------------- main API ----------------

    def compute(self, values: Iterable[Any]) -> Any:
        counter = self._Counter()

        for meta in values:
            txt = self._extract_text_from_metadata(meta)
            if not txt:
                continue

            years = self._extract_years_from_text(txt)
            for y in years:
                counter[y] += 1

        return dict(counter)