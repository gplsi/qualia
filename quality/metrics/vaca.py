from statistics import mean
from typing import Iterable, Any, List

from quality.metrics.base import BaseMetric
from disvaca import VACADiscriminator


class VACARatioMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "vaca-ratio"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def compute(self, values: Iterable[Any]) -> Any:
        discriminator = VACADiscriminator.load()

        all_counts = []
        error_count = 0
        total_items = 0
        errors = set()

        for value in values:
            if value is None:
                continue

            total_items += 1
            try:
                all_counts.append(discriminator.get_counts(value))
            except Exception as e:
                error_count += 1
                errors.add(str(e))

        if error_count > 0:
            print(
                "[Warning @ VACARatioMetric]:"
                f"Encountered {error_count} errors while processing {total_items} items."
                f"Errors: {', '.join(errors)}"
            )

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
