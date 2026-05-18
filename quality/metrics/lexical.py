from typing import Iterable, Any, List

import tiktoken  # pyright: ignore[reportMissingImports]

from quality.metrics.base import BaseMetric


class STTRMetric(BaseMetric):
    """
    Calculates the Standardised Type-Token Ratio (STTR) to measure lexical diversity.
    It mitigates text-length bias by calculating TTR over fixed-size token chunks.
    """

    def __init__(self, chunk_size: int = 1000, encoding_name: str = "cl100k_base"):
        """
        Args:
            chunk_size (int): The number of tokens per standard chunk.
                              1,000 is the standard size used in linguistics.
            encoding_name (str): The tiktoken encoding to use for tokenization.
        """
        self.chunk_size = chunk_size
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return "standardised-type-token-ratio"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _tokenize(self, text: str) -> List[int]:
        return self._encoding.encode(text.lower())

    def compute(self, values: Iterable[Any]) -> float:
        chunk_ttrs: List[float] = []
        current_chunk_tokens: List[int] = []

        for text in values:
            if not isinstance(text, str):
                continue

            tokens = self._tokenize(text)

            for token in tokens:
                current_chunk_tokens.append(token)

                # When the buffer hits the chunk_size, compute TTR and reset
                if len(current_chunk_tokens) == self.chunk_size:
                    unique_types = len(set(current_chunk_tokens))
                    ttr = unique_types / self.chunk_size
                    chunk_ttrs.append(ttr)
                    current_chunk_tokens.clear()  # Reset buffer for next chunk

        # Standard STTR usually discards the final incomplete chunk.
        # However, if the entire corpus is smaller than one chunk,
        # we calculate TTR for whatever tokens we collected to avoid a 0.0 return.
        if len(chunk_ttrs) == 0 and len(current_chunk_tokens) > 0:
            unique_types = len(set(current_chunk_tokens))
            ttr = unique_types / len(current_chunk_tokens)
            chunk_ttrs.append(ttr)

        # If the iterable was completely empty or contained no words
        if not chunk_ttrs:
            return 0.0

        # Return the average TTR across all chunks
        return sum(chunk_ttrs) / len(chunk_ttrs)


class MTLDMetric(BaseMetric):
    """
    Calculates the Measure of Textual Lexical Diversity (MTLD).
    It measures the average number of tokens required to maintain a TTR of 0.72.
    """

    def __init__(self, ttr_threshold: float = 0.72, encoding_name: str = "cl100k_base"):
        """
        Args:
            ttr_threshold (float): The TTR value at which a segment is considered complete.
                                   0.72 is the standard linguistic threshold.
            encoding_name (str): The tiktoken encoding to use for tokenization.
        """
        self.ttr_threshold = ttr_threshold
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def name(self) -> str:
        return "measure-of-textual-lexical-diversity"

    @property
    def target_types(self) -> List[str]:
        return ["text"]

    def _tokenize(self, text: str) -> List[int]:
        return self._encoding.encode(text.lower())

    def _calculate_pass(self, tokens: List[int]) -> float:
        """Helper to calculate MTLD for a single directional pass."""
        if not tokens:
            return 0.0

        factors = 0.0
        types = set()
        token_count = 0

        for token in tokens:
            token_count += 1
            types.add(token)
            ttr = len(types) / token_count

            if ttr < self.ttr_threshold:
                factors += 1.0
                # Reset segment counters
                token_count = 0
                types = set()

        # Handle the final, incomplete segment
        if token_count > 0:
            final_ttr = len(types) / token_count
            # Partial factor calculation: represents how close it was to the threshold
            partial_factor = (1.0 - final_ttr) / (1.0 - self.ttr_threshold)
            factors += partial_factor

        # Failsafe: if text is perfectly diverse (TTR never dropped, factors = 0)
        if factors == 0:
            return float(len(tokens))

        return len(tokens) / factors

    def compute(self, values: Iterable[Any]) -> float:
        document_mtlds: List[float] = []

        for text in values:
            if not isinstance(text, str) or not text.strip():
                continue

            tokens = self._tokenize(text)

            if not tokens:
                continue

            # Bidirectional passes
            forward_mtld = self._calculate_pass(tokens)
            backward_mtld = self._calculate_pass(tokens[::-1])

            # Document MTLD is the average of the forward and backward passes
            doc_mtld = (forward_mtld + backward_mtld) / 2.0
            document_mtlds.append(doc_mtld)

        if not document_mtlds:
            return 0.0

        # Return the average MTLD across all valid text rows
        return sum(document_mtlds) / len(document_mtlds)
