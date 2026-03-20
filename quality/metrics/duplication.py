from typing import Iterable, Any, List

from quality.metrics.base import BaseMetric


class MinhashDuplicationMetric(BaseMetric):
    """Approximate duplication measurement using a MinHash-style sketch.

    Interprets each text item as a set of character shingles (n-grams) and
    maintains a running MinHash signature over the union of all items.
    For every incoming item, it:
      - builds its own MinHash signature
      - estimates its Jaccard similarity to the global union so far
    The final duplication score is the mean of these per-item similarities.

    The score is in [0, 1], where higher values mean more duplication.
    This is a coarse global indicator (not exact pairwise dedup).
    """

    def __init__(
        self,
        window_size: int = 5,
        lowercase: bool = True,
        num_hashes: int = 64,
        seed: int = 42,
    ) -> None:
        self._window_size = max(1, int(window_size))
        self._lowercase = bool(lowercase)
        self._num_hashes = max(1, int(num_hashes))
        self._seed = int(seed)
        self._hash_params = self._make_hash_params()

    @property
    def name(self) -> str:
        return "minhash-duplication"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    # ------------ internal helpers ------------

    def _make_hash_params(self):
        """Pre-generate (a, b, mod) tuples for simple universal hashes."""
        import random

        rng = random.Random(self._seed)
        # large prime > typical 32-bit range
        mod = 2_147_483_647
        params = []
        for _ in range(self._num_hashes):
            a = rng.randrange(1, mod - 1)
            b = rng.randrange(0, mod - 1)
            params.append((a, b, mod))
        return params

    def _shingles(self, text: str):
        """Yield character-based shingles of length `window_size`."""
        if self._lowercase:
            text = text.lower()
        w = self._window_size
        if len(text) < w:
            if text:
                yield text
            return
        for i in range(len(text) - w + 1):
            yield text[i : i + w]

    def _minhash_signature(self, text: str):
        """Compute MinHash signature (list[int]) for one text."""
        import mmh3  # requires python-mmh3 or similar library

        # Initialize with +inf
        sig = [2**63 - 1] * self._num_hashes
        for sh in self._shingles(text):
            # base hash for the shingle
            h = mmh3.hash(sh, self._seed, signed=False)
            for i, (a, b, mod) in enumerate(self._hash_params):
                hv = (a * h + b) % mod
                if hv < sig[i]:
                    sig[i] = hv
        return sig

    @staticmethod
    def _jaccard_from_sigs(sig1, sig2) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = 0
        for a, b in zip(sig1, sig2):
            if a == b:
                matches += 1
        return matches / len(sig1)

    def compute(self, values: Iterable[Any]) -> Any:
        """Stream over values, maintain global MinHash, and aggregate similarity."""
        import math

        global_sig = None  # MinHash over union of all seen items
        sims = []

        for v in values:
            if v is None:
                continue
            text = str(v).strip()
            if not text:
                continue

            sig = self._minhash_signature(text)
            # If no shingles produced (e.g. empty), skip
            if all(h == 2**63 - 1 for h in sig):
                continue

            if global_sig is None:
                # First document initializes global signature; similarity is 1
                global_sig = sig[:]
                sims.append(1.0)
                continue

            # similarity of this document to global union so far
            sim = self._jaccard_from_sigs(sig, global_sig)
            sims.append(sim)

            # Update global signature = element-wise min (union sketch)
            for i in range(len(global_sig)):
                if sig[i] < global_sig[i]:
                    global_sig[i] = sig[i]

        if not sims:
            return 0.0

        # Mean similarity as duplication indicator
        return float(sum(sims) / len(sims))
