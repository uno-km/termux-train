"""
termux_train/tokenization/word.py
=================================
Whitespace and Punctuation Preserving Word-Level Tokenizer (Lossless Regex Lexer).

Uses regex lexing to treat words, whitespace spans, and punctuation as discrete tokens.
Guarantees exact whitespace/layout preservation and deterministic vocabulary ordering.
Note: This is a layout-preserving regex lexer, not a linguistic morphological analyzer.
"""

import re
from typing import List, Sequence, Optional, Dict, Any
from collections import Counter
from .base import BaseTokenizer

# Lexer pattern: matches contiguous whitespace, individual punctuation marks, or word characters
TOKEN_PATTERN = re.compile(r"(\s+|[^\w\s]|\w+)")
LEXER_NAME = "unicode_word_whitespace_punctuation"
LEXER_VERSION = "1"


class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer with whitespace and punctuation preservation.

    Unlike naive split() tokenizers, this tokenizer preserves all spaces, tabs,
    newlines, and punctuation marks so that known-vocabulary text can be reconstructed exactly.
    """

    LEXER_NAME = LEXER_NAME
    LEXER_VERSION = LEXER_VERSION

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
        if len(texts) == 0:
            raise ValueError("Cannot build vocabulary from an empty texts sequence")
        if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq < 1:
            raise ValueError(f"min_freq must be an integer >= 1, got {min_freq}")
        if max_vocab_size is not None:
            if isinstance(max_vocab_size, bool) or not isinstance(max_vocab_size, int) or max_vocab_size < len(self.SPECIAL_TOKENS):
                raise ValueError(
                    f"max_vocab_size must be >= {len(self.SPECIAL_TOKENS)} (minimum for special tokens), got {max_vocab_size}"
                )

        self._init_special_tokens()

        counts = Counter()
        total_tokens = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError(f"All elements in texts must be strings, got {type(text).__name__}")
            tokens = self._tokenize(text)
            counts.update(tokens)
            total_tokens += len(tokens)

        if total_tokens == 0:
            raise ValueError("Cannot build vocabulary from texts containing zero tokens")

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

    def get_config(self) -> Dict[str, Any]:
        return {
            "lexer": self.LEXER_NAME,
            "lexer_version": self.LEXER_VERSION,
        }

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")
        if config.get("lexer") != self.LEXER_NAME:
            raise ValueError(f"Unsupported lexer: expected '{self.LEXER_NAME}', got '{config.get('lexer')}'")
        if config.get("lexer_version") != self.LEXER_VERSION:
            raise ValueError(
                f"Unsupported lexer_version: expected '{self.LEXER_VERSION}', got '{config.get('lexer_version')}'"
            )
