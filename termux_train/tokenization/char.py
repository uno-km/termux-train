"""
termux_train/tokenization/char.py
=================================
Character-Level Lightweight Tokenizer.

Treats individual Unicode characters as discrete tokens without lossy normalization.
Guarantees deterministic vocabulary ordering and exact round-trip for known characters.
"""

from typing import List, Sequence, Optional, Dict, Any
from collections import Counter
from .base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.

    Each character (ASCII, Unicode code point, whitespace, newline) is a token.
    Preserves exact code-point sequences without implicit Unicode normalization.
    """

    def build_vocab(
        self,
        texts: Sequence[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> "CharTokenizer":
        """
        Builds a character vocabulary deterministically from the given texts.

        Tie-breaking rule: Higher frequency first; alphabetical order for equal frequencies.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"texts must be a list or tuple of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            raise ValueError("Cannot build vocabulary from an empty texts sequence")
        if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError(f"min_freq must be an integer >= 1, got {min_freq}")
        if max_vocab_size is not None:
            if isinstance(max_vocab_size, bool) or not isinstance(max_vocab_size, int) or max_vocab_size < len(self.SPECIAL_TOKENS):
                raise ValueError(
                    f"max_vocab_size must be >= {len(self.SPECIAL_TOKENS)} (minimum for special tokens), got {max_vocab_size}"
                )

        # Reset vocabulary with special tokens
        self._init_special_tokens()

        counts = Counter()
        total_chars = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"All elements in texts must be strings, got {type(text).__name__}")
            counts.update(list(text))
            total_chars += len(text)

        if total_chars == 0:
            raise ValueError("Cannot build vocabulary from texts containing zero characters")

        # Filter by minimum frequency and sort deterministically: (-count, char)
        sorted_chars = sorted(
            [char for char, freq in counts.items() if freq >= min_freq and char not in self._token_to_id],
            key=lambda c: (-counts[c], c),
        )

        if max_vocab_size is not None:
            available_slots = max_vocab_size - len(self._token_to_id)
            sorted_chars = sorted_chars[:max(0, available_slots)]

        for char in sorted_chars:
            idx = len(self._token_to_id)
            self._token_to_id[char] = idx
            self._id_to_token[idx] = char

        self._vocab_built = True
        return self

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into individual characters."""
        return list(text)

    def _detokenize(self, token_strings: List[str]) -> str:
        """Concatenates characters directly."""
        return "".join(token_strings)

    def get_config(self) -> Dict[str, Any]:
        return {}

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")
