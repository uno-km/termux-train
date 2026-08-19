"""
termux_train/tokenization/word.py
=================================
Whitespace and Punctuation Preserving Word-Level Tokenizer.

Uses regex lexing to treat words, whitespace spans, and punctuation as discrete tokens.
Guarantees exact whitespace/layout preservation and deterministic vocabulary ordering.
"""

import re
from typing import List, Sequence, Optional, Dict
from collections import Counter
from .base import BaseTokenizer

# Lexer pattern: matches contiguous whitespace, individual punctuation marks, or word characters
TOKEN_PATTERN = re.compile(r"(\s+|[^\w\s]|\w+)")


class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer with whitespace and punctuation preservation.

    Unlike naive split() tokenizers, this tokenizer preserves all spaces, tabs,
    newlines, and punctuation marks so that known-vocabulary text can be reconstructed exactly.
    """

    def build_vocab(
        self,
        texts: Sequence[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> "WordTokenizer":
        """
        Builds a word vocabulary deterministically from the given texts.

        Tie-breaking rule: Higher frequency first; alphabetical order for equal frequencies.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError(f"texts must be a list or tuple of strings, got {type(texts).__name__}")
        if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError(f"min_freq must be an integer >= 1, got {min_freq}")
        if max_vocab_size is not None:
            if isinstance(max_vocab_size, bool) or not isinstance(max_vocab_size, int) or max_vocab_size < len(self.SPECIAL_TOKENS):
                raise ValueError(
                    f"max_vocab_size must be >= {len(self.SPECIAL_TOKENS)} (minimum for special tokens), got {max_vocab_size}"
                )

        self._init_special_tokens()

        counts = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"All elements in texts must be strings, got {type(text).__name__}")
            tokens = self._tokenize(text)
            counts.update(tokens)

        # Filter and sort deterministically: (-count, token)
        sorted_tokens = sorted(
            [t for t, freq in counts.items() if freq >= min_freq and t not in self._token_to_id],
            key=lambda t: (-counts[t], t),
        )

        if max_vocab_size is not None:
            available_slots = max_vocab_size - len(self._token_to_id)
            sorted_tokens = sorted_tokens[:max(0, available_slots)]

        for token in sorted_tokens:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token

        self._vocab_built = True
        return self

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into words, punctuation marks, and whitespace spans."""
        if not text:
            return []
        matches = TOKEN_PATTERN.findall(text)
        return matches

    def _detokenize(self, token_strings: List[str]) -> str:
        """Concatenates tokens directly (preserving whitespace tokens)."""
        return "".join(token_strings)
