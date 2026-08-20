"""
termux_train/tokenization/base.py
=================================
Abstract Base Tokenizer Interface, Strict Schema Validation, and Special Token Contracts.

Defines deterministic vocabulary management and encode/decode lifecycle.
Pure Python only (zero external / C++ / Rust / Tensor / NumPy dependencies).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Sequence, Optional, Any
import copy

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
FORMAT_NAME = "termux-train-tokenizer"
SCHEMA_VERSION = "1.0"


class BaseTokenizer(ABC):
    """
    Abstract base class for lightweight tokenizers in termux-train.

    Guarantees:
      - Deterministic vocabulary indexing.
      - Fixed special token mapping:
          <PAD> -> 0, <UNK> -> 1, <BOS> -> 2, <EOS> -> 3.
      - Pure-Python execution (no C++ extensions or third-party tokenizers).
      - Strict serialization schema validation (format, version, type, contiguity, and invariants).
    """

    PAD_TOKEN = PAD_TOKEN
    UNK_TOKEN = UNK_TOKEN
    BOS_TOKEN = BOS_TOKEN
    EOS_TOKEN = EOS_TOKEN

    PAD_ID = PAD_ID
    UNK_ID = UNK_ID
    BOS_ID = BOS_ID
    EOS_ID = EOS_ID
    SPECIAL_TOKENS = SPECIAL_TOKENS

    FORMAT_NAME = FORMAT_NAME
    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self) -> None:
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._vocab_built: bool = False
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initializes the fixed special token mappings."""
        self._token_to_id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            BOS_TOKEN: BOS_ID,
            EOS_TOKEN: EOS_ID,
        }
        self._id_to_token = {
            PAD_ID: PAD_TOKEN,
            UNK_ID: UNK_TOKEN,
            BOS_ID: BOS_TOKEN,
            EOS_ID: EOS_TOKEN,
        }

    @property
    def vocab_size(self) -> int:
        """Returns the current number of unique tokens in the vocabulary."""
        return len(self._token_to_id)

    @property
    def is_built(self) -> bool:
        """Returns True if the vocabulary has been constructed."""
        return self._vocab_built

    def token_to_id(self, token: str) -> int:
        """
        Maps a token string to its integer ID.
        Returns UNK_ID if token is not found in the vocabulary.
        """
        if not isinstance(token, str):
            raise TypeError(f"token must be a str, got {type(token).__name__}")
        return self._token_to_id.get(token, self.UNK_ID)

    def id_to_token(self, token_id: int) -> str:
        """
        Maps an integer token ID to its string representation.
        Raises KeyError if the token ID is out of range.
        """
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"token_id must be an int, got {type(token_id).__name__}")
        if token_id not in self._id_to_token:
            raise KeyError(f"Token ID {token_id} not found in vocabulary (vocab_size={self.vocab_size})")
        return self._id_to_token[token_id]

    def get_vocab(self) -> Dict[str, int]:
        """Returns a copy of the vocabulary dictionary mapping tokens to IDs."""
        return copy.deepcopy(self._token_to_id)

    def get_config(self) -> Dict[str, Any]:
        """Returns subclass-specific configuration parameters for serialization."""
        return {}

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Hook for subclass-specific configuration validation during deserialization."""
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

    @abstractmethod
    def build_vocab(
        self,
        texts: Sequence[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> "BaseTokenizer":
        """
        Constructs the vocabulary deterministically from the given texts.

        Args:
            texts: Sequence of training strings.
            min_freq: Minimum occurrence count for a token to be included.
            max_vocab_size: Maximum total vocabulary size including special tokens.

        Returns:
            self
        """
        pass

    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """Splits raw text into a list of token strings."""
        pass

    @abstractmethod
    def _detokenize(self, token_strings: List[str]) -> str:
        """Combines a list of token strings into a reconstructed text."""
        pass

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encodes a string into a list of integer token IDs.

        Args:
            text: Input string to tokenize.
            add_bos: Whether to prepend BOS_ID at the start.
            add_eos: Whether to append EOS_ID at the end.

        Returns:
            List[int] of token IDs.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not isinstance(add_bos, bool):
            raise TypeError(f"add_bos must be a bool, got {type(add_bos).__name__}")
        if not isinstance(add_eos, bool):
            raise TypeError(f"add_eos must be a bool, got {type(add_eos).__name__}")
        if not self._vocab_built:
            raise RuntimeError("Tokenizer vocabulary has not been built yet. Call build_vocab() first.")

        token_strs = self._tokenize(text)
        ids = [self.token_to_id(t) for t in token_strs]

        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decodes a sequence of integer token IDs back into a string.

        Args:
            tokens: Sequence of integer token IDs.
            skip_special_tokens: If True (default), filters out <PAD>, <UNK>, <BOS>, <EOS>.

        Returns:
            Reconstructed string.
        """
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(f"tokens must be a list or tuple of ints, got {type(tokens).__name__}")
        if not isinstance(skip_special_tokens, bool):
            raise TypeError(f"skip_special_tokens must be a bool, got {type(skip_special_tokens).__name__}")
        if not self._vocab_built:
            raise RuntimeError("Tokenizer vocabulary has not been built yet. Call build_vocab() first.")

        token_strings = []
        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        for idx, t in enumerate(tokens):
            if isinstance(t, bool) or not isinstance(t, int):
                raise TypeError(f"Token at index {idx} must be an integer, got {type(t).__name__}")
            if t < 0 or t >= self.vocab_size:
                raise ValueError(f"Token ID {t} at index {idx} is out of bounds for vocab of size {self.vocab_size}")

            if skip_special_tokens and t in special_ids:
                continue

            token_strings.append(self._id_to_token[t])

        return self._detokenize(token_strings)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the tokenizer vocabulary and metadata to a strict JSON-compatible container.
        """
        return {
            "format": self.FORMAT_NAME,
            "version": self.SCHEMA_VERSION,
            "tokenizer_type": self.__class__.__name__,
            "vocab": self.get_vocab(),
            "vocab_built": self._vocab_built,
            "config": self.get_config(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseTokenizer":
        """
        Atomically deserializes and strictly validates a tokenizer container.

        Enforces:
          - Exact container key set.
          - Exact format name and schema version.
          - Strict tokenizer_type class matching.
          - Strict vocab type, string keys, integer non-bool IDs >= 0.
          - Strict ID uniqueness and contiguity from 0 to vocab_size - 1.
          - Exact special token ID mappings (<PAD>:0, <UNK>:1, <BOS>:2, <EOS>:3).
          - Subclass-specific configuration validation.
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")

        expected_keys = {"format", "version", "tokenizer_type", "vocab", "vocab_built", "config"}
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"Invalid tokenizer container keys: expected {sorted(expected_keys)}, got {sorted(actual_keys)}")

        if data["format"] != cls.FORMAT_NAME:
            raise ValueError(f"Unsupported tokenizer format: {data['format']}, expected {cls.FORMAT_NAME}")

        if data["version"] != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported tokenizer schema version: {data['version']}, expected {cls.SCHEMA_VERSION}")

        if data["tokenizer_type"] != cls.__name__:
            raise ValueError(
                f"Tokenizer type mismatch: container contains '{data['tokenizer_type']}', but target class is '{cls.__name__}'"
            )

        vocab = data["vocab"]
        if not isinstance(vocab, dict):
            raise TypeError(f"vocab must be a dict, got {type(vocab).__name__}")

        vocab_built = data["vocab_built"]
        if not isinstance(vocab_built, bool):
            raise TypeError(f"vocab_built must be a bool, got {type(vocab_built).__name__}")

        config = data["config"]
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")

        # Validate vocabulary keys and integer IDs
        seen_ids = set()
        for token_key, token_val in vocab.items():
            if not isinstance(token_key, str):
                raise TypeError(f"All vocabulary token keys must be str, got {type(token_key).__name__}")
            if isinstance(token_val, bool) or not isinstance(token_val, int):
                raise TypeError(f"Vocabulary ID for token '{token_key}' must be an int, got {type(token_val).__name__}")
            if token_val < 0:
                raise ValueError(f"Vocabulary ID for token '{token_key}' must be >= 0, got {token_val}")
            if token_val in seen_ids:
                raise ValueError(f"Duplicate vocabulary ID {token_val} found in serialized vocab")
            seen_ids.add(token_val)

        # Validate contiguity: IDs must be 0..len(vocab)-1
        num_tokens = len(vocab)
        if seen_ids != set(range(num_tokens)):
            raise ValueError(
                f"Vocabulary IDs must be strictly contiguous from 0 to {num_tokens - 1}, but found missing or out-of-range IDs"
            )

        # Validate fixed special token IDs
        if vocab.get(PAD_TOKEN) != PAD_ID:
            raise ValueError(f"Special token '{PAD_TOKEN}' must have ID {PAD_ID}, got {vocab.get(PAD_TOKEN)}")
        if vocab.get(UNK_TOKEN) != UNK_ID:
            raise ValueError(f"Special token '{UNK_TOKEN}' must have ID {UNK_ID}, got {vocab.get(UNK_TOKEN)}")
        if vocab.get(BOS_TOKEN) != BOS_ID:
            raise ValueError(f"Special token '{BOS_TOKEN}' must have ID {BOS_ID}, got {vocab.get(BOS_TOKEN)}")
        if vocab.get(EOS_TOKEN) != EOS_ID:
            raise ValueError(f"Special token '{EOS_TOKEN}' must have ID {EOS_ID}, got {vocab.get(EOS_TOKEN)}")

        instance = cls()
        instance._validate_config(config)

        # Atomic assignment after all validations pass
        instance._token_to_id = copy.deepcopy(vocab)
        instance._id_to_token = {v: k for k, v in instance._token_to_id.items()}
        instance._vocab_built = vocab_built

        return instance
