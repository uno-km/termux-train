"""
termux_train/tokenization/byte.py
=================================
Byte-Level Lightweight Tokenizer.

Operates directly on UTF-8 raw byte values (0~255) with fixed vocabulary indexing.
Guarantees exact round-trip representation for all valid UTF-8 strings.
"""

from typing import List, Sequence, Optional, Dict, Any
from .base import BaseTokenizer


class ByteTokenizer(BaseTokenizer):
    """
    Byte-level UTF-8 tokenizer.

    Maps each byte value (0..255) to a fixed token ID:
      - 0: <PAD>
      - 1: <UNK>
      - 2: <BOS>
      - 3: <EOS>
      - 4..259: Byte values 0..255 (formatted as '<0x00>' .. '<0xFF>')
    """

    BYTE_OFFSET = 4  # Reserve 0..3 for special tokens
    NUM_BYTES = 256

    def __init__(self) -> None:
        super().__init__()
        # Auto-build full 256-byte vocabulary
        self.build_vocab()

    def build_vocab(
        self,
        texts: Optional[Sequence[str]] = None,
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None,
    ) -> "ByteTokenizer":
        """
        Populates the full 256-byte vocabulary.

        Args:
            texts: Optional sequence of strings (validated for type consistency if provided).
            min_freq: Must be 1 (ByteTokenizer vocabulary is fixed).
            max_vocab_size: Must be None or 260 (ByteTokenizer vocabulary size is fixed).
        """
        if texts is not None:
            if not isinstance(texts, (list, tuple)):
                raise TypeError(f"texts must be a list or tuple of strings or None, got {type(texts).__name__}")
            for t in texts:
                if not isinstance(t, str):
                    raise TypeError(f"All elements in texts must be strings, got {type(t).__name__}")

        if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq != 1:
            raise ValueError(f"ByteTokenizer uses a fixed 256-byte vocabulary; min_freq must be 1, got {min_freq}")

        if max_vocab_size is not None:
            if isinstance(max_vocab_size, bool) or not isinstance(max_vocab_size, int) or max_vocab_size != 260:
                raise ValueError(
                    f"ByteTokenizer uses a fixed 260-token vocabulary; max_vocab_size must be None or 260, got {max_vocab_size}"
                )

        self._init_special_tokens()

        for b in range(self.NUM_BYTES):
            token_str = f"<0x{b:02X}>"
            token_id = b + self.BYTE_OFFSET
            self._token_to_id[token_str] = token_id
            self._id_to_token[token_id] = token_str

        self._vocab_built = True
        return self

    def _tokenize(self, text: str) -> List[str]:
        """Converts UTF-8 text into byte token strings."""
        raw_bytes = text.encode("utf-8")
        return [f"<0x{b:02X}>" for b in raw_bytes]

    def _detokenize(self, token_strings: List[str]) -> str:
        """Detokenization handled directly in decode() via byte spans for UTF-8 preservation."""
        return "".join(token_strings)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encodes UTF-8 string into byte token IDs.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if not isinstance(add_bos, bool):
            raise TypeError(f"add_bos must be a bool, got {type(add_bos).__name__}")
        if not isinstance(add_eos, bool):
            raise TypeError(f"add_eos must be a bool, got {type(add_eos).__name__}")

        raw_bytes = text.encode("utf-8")
        ids = [b + self.BYTE_OFFSET for b in raw_bytes]

        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(
        self,
        tokens: Sequence[int],
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        """
        Decodes byte token IDs back into a UTF-8 string.

        Args:
            tokens: Sequence of integer token IDs.
            skip_special_tokens: If True (default), filters out special tokens (<PAD>, <UNK>, <BOS>, <EOS>).
                                 If False, inserts special token string representations (<BOS>, <EOS>, etc.)
                                 in place while decoding intermediate byte sequences to UTF-8.
            errors: UTF-8 decode error handling policy ("strict", "replace", "ignore").

        Returns:
            Reconstructed UTF-8 string.
        """
        if not isinstance(tokens, (list, tuple)):
            raise TypeError(f"tokens must be a list or tuple of ints, got {type(tokens).__name__}")
        if not isinstance(skip_special_tokens, bool):
            raise TypeError(f"skip_special_tokens must be a bool, got {type(skip_special_tokens).__name__}")
        if not isinstance(errors, str):
            raise TypeError(f"errors must be a str, got {type(errors).__name__}")

        output_parts: List[str] = []
        current_byte_span = bytearray()
        special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        def flush_bytes():
            if current_byte_span:
                try:
                    output_parts.append(current_byte_span.decode("utf-8", errors=errors))
                except UnicodeDecodeError as e:
                    raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end, f"Failed to decode byte sequence: {e.reason}") from e
                current_byte_span.clear()

        for idx, t in enumerate(tokens):
            if isinstance(t, bool) or not isinstance(t, int):
                raise TypeError(f"Token at index {idx} must be an integer, got {type(t).__name__}")
            if t < 0 or t >= self.vocab_size:
                raise ValueError(f"Token ID {t} at index {idx} is out of bounds for vocab of size {self.vocab_size}")

            if t in special_ids:
                flush_bytes()
                if not skip_special_tokens:
                    output_parts.append(self._id_to_token[t])
            else:
                current_byte_span.append(t - self.BYTE_OFFSET)

        flush_bytes()
        return "".join(output_parts)

    def get_config(self) -> Dict[str, Any]:
        return {
            "byte_offset": self.BYTE_OFFSET,
            "num_bytes": self.NUM_BYTES,
        }

    def _validate_config(self, config: Dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")
        if config.get("byte_offset") != self.BYTE_OFFSET:
            raise ValueError(f"Invalid byte_offset: expected {self.BYTE_OFFSET}, got {config.get('byte_offset')}")
        if config.get("num_bytes") != self.NUM_BYTES:
            raise ValueError(f"Invalid num_bytes: expected {self.NUM_BYTES}, got {config.get('num_bytes')}")
