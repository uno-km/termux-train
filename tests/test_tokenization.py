"""
tests/test_tokenization.py
==========================
Comprehensive Test Suite for SCRUM-313: Lightweight Pure-Python Tokenizers.

Tests:
  1. BaseTokenizer interface, special tokens, and unbuilt error handling
  2. CharTokenizer deterministic vocabulary, tie-breaking, and exact round-trip
  3. CharTokenizer unknown character replacement and Unicode code-point preservation
  4. CharTokenizer BOS / EOS special token handling and filtering
  5. ByteTokenizer 260-token vocabulary, exact UTF-8 round-trip, and error policies
  6. WordTokenizer whitespace and punctuation preservation and layout retention
  7. WordTokenizer deterministic vocab, min_freq, and max_vocab_size
  8. Empty string, special literal collisions, and invalid input type rejections
  9. Tokenizer serialization round-trip (to_dict / from_dict)
  10. Pure Python contract (no Tensor / NumPy dependency)
"""

import pytest
from termux_train.tokenization import (
    BaseTokenizer,
    CharTokenizer,
    ByteTokenizer,
    WordTokenizer,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
)


# =============================================================================
# Section 1: BaseTokenizer Interface and Contracts
# =============================================================================

def test_base_tokenizer_special_tokens_and_constants():
    assert PAD_TOKEN == "<PAD>"
    assert UNK_TOKEN == "<UNK>"
    assert BOS_TOKEN == "<BOS>"
    assert EOS_TOKEN == "<EOS>"

    assert PAD_ID == 0
    assert UNK_ID == 1
    assert BOS_ID == 2
    assert EOS_ID == 3


def test_base_tokenizer_unbuilt_error_handling():
    tok = CharTokenizer()
    assert tok.is_built is False

    # Calling encode before build_vocab raises RuntimeError
    with pytest.raises(RuntimeError, match="not been built yet"):
        tok.encode("hello")

    # Calling decode before build_vocab raises RuntimeError
    with pytest.raises(RuntimeError, match="not been built yet"):
        tok.decode([0, 1])


def test_base_tokenizer_invalid_input_types():
    tok = CharTokenizer()
    tok.build_vocab(["abc"])

    # encode non-string inputs
    for bad_input in [123, None, ["a"], {"a": 1}, True]:
        with pytest.raises(TypeError, match="text must be a str"):
            tok.encode(bad_input)

    # encode invalid add_bos / add_eos types
    with pytest.raises(TypeError, match="add_bos must be a bool"):
        tok.encode("a", add_bos=1)
    with pytest.raises(TypeError, match="add_eos must be a bool"):
        tok.encode("a", add_eos="True")

    # decode non-sequence inputs
    for bad_seq in [123, "123", None, 3.14]:
        with pytest.raises(TypeError, match="tokens must be a list or tuple"):
            tok.decode(bad_seq)

    # decode invalid token ID types
    for bad_id in [True, False, 1.5, "1", None]:
        with pytest.raises(TypeError, match="must be an integer"):
            tok.decode([0, bad_id])

    # decode out-of-bounds token IDs
    with pytest.raises(ValueError, match="out of bounds"):
        tok.decode([-1])
    with pytest.raises(ValueError, match="out of bounds"):
        tok.decode([9999])


def test_token_to_id_and_id_to_token_lookups():
    tok = CharTokenizer()
    tok.build_vocab(["xyz"])

    assert tok.token_to_id("x") > 3
    assert tok.token_to_id("unknown_symbol") == UNK_ID
    assert tok.token_to_id(PAD_TOKEN) == PAD_ID

    with pytest.raises(TypeError, match="token must be a str"):
        tok.token_to_id(123)

    x_id = tok.token_to_id("x")
    assert tok.id_to_token(x_id) == "x"
    assert tok.id_to_token(BOS_ID) == BOS_TOKEN

    with pytest.raises(KeyError, match="not found in vocabulary"):
        tok.id_to_token(9999)


# =============================================================================
# Section 2: CharTokenizer Deterministic Vocab & Exact Round-Trip
# =============================================================================

def test_char_tokenizer_deterministic_vocab_and_tie_breaking():
    # 'b' appears 3 times, 'a' appears 2 times, 'c' appears 2 times ('a' < 'c' lexically)
    texts = ["b b b a a c c"]
    tok1 = CharTokenizer().build_vocab(texts)
    tok2 = CharTokenizer().build_vocab(texts)

    # Exact vocabulary parity across instances
    assert tok1.get_vocab() == tok2.get_vocab()

    # 'b' has higher frequency -> assigned smaller ID than 'a'
    assert tok1.token_to_id("b") < tok1.token_to_id("a")
    # 'a' and 'c' have same frequency -> 'a' before 'c' alphabetically
    assert tok1.token_to_id("a") < tok1.token_to_id("c")


def test_char_tokenizer_min_freq_and_max_vocab_size():
    corpus = ["apple banana cherry date elderberry"]
    # min_freq = 3 filters rare characters
    tok = CharTokenizer().build_vocab(corpus, min_freq=3)
    # 'e' and 'a' and ' ' appear >= 3 times
    assert tok.token_to_id("e") != UNK_ID
    assert tok.token_to_id("a") != UNK_ID
    # 'z' does not exist
    assert tok.token_to_id("z") == UNK_ID

    # max_vocab_size limits total vocabulary including 4 special tokens
    tok_limited = CharTokenizer().build_vocab(corpus, max_vocab_size=6)
    assert tok_limited.vocab_size == 6


def test_char_tokenizer_known_vocab_exact_round_trip():
    text = "Hello, World!\nThis is a Termux-Train CharTokenizer test 123.\tTabs and symbols: @#$%^&*()_+"
    tok = CharTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text
    assert len(encoded) == len(text)
    assert UNK_ID not in encoded


def test_char_tokenizer_unknown_character_policy():
    tok = CharTokenizer().build_vocab(["abc"])
    text = "a x b y c"
    encoded = tok.encode(text)

    # 'x' and 'y' and spaces are unknown
    assert encoded[0] == tok.token_to_id("a")
    assert encoded[1] == UNK_ID  # space
    assert encoded[2] == UNK_ID  # 'x'

    decoded = tok.decode(encoded)
    assert decoded == "a<UNK>b<UNK>c" or UNK_TOKEN in decoded


def test_char_tokenizer_bos_eos_options():
    text = "train"
    tok = CharTokenizer().build_vocab([text])

    # No BOS/EOS
    ids_plain = tok.encode(text, add_bos=False, add_eos=False)
    assert ids_plain[0] != BOS_ID
    assert ids_plain[-1] != EOS_ID

    # With BOS and EOS
    ids_tagged = tok.encode(text, add_bos=True, add_eos=True)
    assert ids_tagged[0] == BOS_ID
    assert ids_tagged[-1] == EOS_ID
    assert ids_tagged[1:-1] == ids_plain

    # Decode skipping special tokens restores original text
    assert tok.decode(ids_tagged, skip_special_tokens=True) == text
    # Decode keeping special tokens includes <BOS> and <EOS>
    assert tok.decode(ids_tagged, skip_special_tokens=False) == f"<BOS>{text}<EOS>"


def test_char_tokenizer_unicode_multilingual_and_emoji():
    # Multilingual without auto-normalization
    multilingual = "안녕하세요! 🚀 딥러닝 PyTorch Termux-Train (株) Café 123"
    tok = CharTokenizer().build_vocab([multilingual])

    encoded = tok.encode(multilingual)
    decoded = tok.decode(encoded)

    assert decoded == multilingual
    assert UNK_ID not in encoded


# =============================================================================
# Section 3: ByteTokenizer 260-Token Vocab & Exact UTF-8 Round-Trip
# =============================================================================

def test_byte_tokenizer_auto_built_vocab_and_size():
    tok = ByteTokenizer()
    assert tok.is_built is True
    assert tok.vocab_size == 260  # 4 special + 256 byte tokens

    assert tok.token_to_id("<0x00>") == 4
    assert tok.token_to_id("<0xFF>") == 259
    assert tok.id_to_token(4) == "<0x00>"
    assert tok.id_to_token(259) == "<0xFF>"


def test_byte_tokenizer_exact_round_trip_utf8():
    corpus = [
        "Simple ASCII text 12345!@#$",
        "한국어 자연어 처리 및 온디바이스 토크나이저 검증",
        "🔥 Multilingual text: 日本語, Español, Français, Deutsch, Русский 🚀",
        "Control characters: \n\t\r\0 and symbols",
    ]

    tok = ByteTokenizer()

    for text in corpus:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text
        assert len(encoded) == len(text.encode("utf-8"))


def test_byte_tokenizer_bos_eos_options():
    tok = ByteTokenizer()
    text = "hello"
    encoded = tok.encode(text, add_bos=True, add_eos=True)

    assert encoded[0] == BOS_ID
    assert encoded[-1] == EOS_ID

    # Decode with skip_special_tokens restores original string
    assert tok.decode(encoded, skip_special_tokens=True) == text


def test_byte_tokenizer_strict_decode_error_handling():
    tok = ByteTokenizer()
    # Create invalid UTF-8 sequence (e.g. solitary continuation byte 0x80)
    invalid_byte_id = 0x80 + ByteTokenizer.BYTE_OFFSET

    with pytest.raises(UnicodeDecodeError):
        tok.decode([invalid_byte_id], errors="strict")

    # errors="replace" produces replacement character '\ufffd'
    res_replaced = tok.decode([invalid_byte_id], errors="replace")
    assert "\ufffd" in res_replaced


# =============================================================================
# Section 4: WordTokenizer Whitespace & Punctuation Preservation
# =============================================================================

def test_word_tokenizer_whitespace_and_punctuation_preservation():
    text = "  Hello,   world! \n\tThis is a   DocFold <DOC> test.\n"
    tok = WordTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    # Exact layout and whitespace preservation
    assert decoded == text
    assert UNK_ID not in encoded


def test_word_tokenizer_deterministic_vocab_and_min_freq():
    corpus = ["apple banana apple cherry apple banana date"]
    tok = WordTokenizer().build_vocab(corpus, min_freq=2)

    # 'apple' (freq 3) -> ID < 'banana' (freq 2)
    assert tok.token_to_id("apple") < tok.token_to_id("banana")
    # 'cherry' and 'date' (freq 1) are filtered out
    assert tok.token_to_id("cherry") == UNK_ID
    assert tok.token_to_id("date") == UNK_ID


def test_word_tokenizer_korean_and_multilingual():
    text = "안녕하세요! 이것은 온디바이스 Transformer 토크나이저 테스트입니다."
    tok = WordTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text


def test_word_tokenizer_unknown_word_policy():
    tok = WordTokenizer().build_vocab(["The quick brown fox"])
    text = "The lazy brown dog"
    encoded = tok.encode(text)

    # 'The' and 'brown' are known, 'lazy' and 'dog' are UNK
    assert encoded[0] == tok.token_to_id("The")
    assert tok.token_to_id("lazy") == UNK_ID
    assert tok.token_to_id("dog") == UNK_ID


# =============================================================================
# Section 5: General Edge Cases and Serialization
# =============================================================================

def test_all_tokenizers_empty_string():
    for TokClass in [CharTokenizer, ByteTokenizer, WordTokenizer]:
        tok = TokClass()
        if not tok.is_built:
            tok.build_vocab(["abc"])

        # Plain empty string
        assert tok.encode("") == []
        assert tok.decode([]) == ""

        # Empty string with BOS/EOS
        assert tok.encode("", add_bos=True, add_eos=True) == [BOS_ID, EOS_ID]
        assert tok.decode([BOS_ID, EOS_ID], skip_special_tokens=True) == ""


def test_special_token_literal_text_handling():
    # When input text literally contains the string "<UNK>" or "<PAD>"
    text = "Value is <UNK> and <PAD>"
    tok = CharTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_tokenizer_serialization_round_trip():
    corpus = ["Hello, world! 123", "Termux-Train Tokenizer"]
    orig_tok = CharTokenizer().build_vocab(corpus)

    data = orig_tok.to_dict()
    restored_tok = CharTokenizer.from_dict(data)

    assert restored_tok.is_built is True
    assert restored_tok.vocab_size == orig_tok.vocab_size
    assert restored_tok.get_vocab() == orig_tok.get_vocab()

    test_str = "Hello, world!"
    assert restored_tok.encode(test_str) == orig_tok.encode(test_str)
    assert restored_tok.decode(orig_tok.encode(test_str)) == test_str


def test_tokenization_pure_python_contract():
    # Verify module has no PyTorch / Torch / C extensions
    import termux_train.tokenization.base as t_base
    import termux_train.tokenization.char as t_char
    import termux_train.tokenization.byte as t_byte
    import termux_train.tokenization.word as t_word

    for mod in [t_base, t_char, t_byte, t_word]:
        assert "torch" not in mod.__dict__
        assert "numpy" not in mod.__dict__
        assert "ctypes" not in mod.__dict__
