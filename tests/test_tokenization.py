"""
tests/test_tokenization.py
==========================
Comprehensive Test Suite for SCRUM-313: Lightweight Pure-Python Tokenizers.

Hardened Coverage:
  1. BaseTokenizer interface, special tokens, and unbuilt error handling
  2. Strict serialization schema validation (P0-1)
  3. Unified special token decode semantics across all subclasses (P0-2)
  4. Subclass config serialization and validation hooks (P0-3)
  5. Isolated subprocess zero-dependency verification (P1-1)
  6. ByteTokenizer strict argument validation (P1-2)
  7. Empty corpus validation across tokenizers (P1-3)
  8. CharTokenizer deterministic vocabulary, tie-breaking, and exact round-trip
  9. ByteTokenizer 260-token vocabulary, exact UTF-8 round-trip, and error policies
  10. WordTokenizer whitespace/punctuation preservation and layout retention
"""

import sys
import subprocess
import copy
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

    with pytest.raises(RuntimeError, match="not been built yet"):
        tok.encode("hello")

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
# Section 2: Strict Serialization Schema Validation (P0-1 & P0-3)
# =============================================================================

def test_tokenizer_serialization_valid_round_trip():
    corpus = ["Hello, world! 123", "Termux-Train Tokenizer"]
    orig_tok = CharTokenizer().build_vocab(corpus)

    data = orig_tok.to_dict()
    assert data["format"] == "termux-train-tokenizer"
    assert data["version"] == "1.0"
    assert data["tokenizer_type"] == "CharTokenizer"
    assert isinstance(data["vocab"], dict)
    assert isinstance(data["config"], dict)

    restored_tok = CharTokenizer.from_dict(data)
    assert restored_tok.is_built is True
    assert restored_tok.vocab_size == orig_tok.vocab_size
    assert restored_tok.get_vocab() == orig_tok.get_vocab()

    test_str = "Hello, world!"
    assert restored_tok.encode(test_str) == orig_tok.encode(test_str)
    assert restored_tok.decode(orig_tok.encode(test_str)) == test_str


def test_tokenizer_serialization_strict_schema_rejections():
    tok = CharTokenizer().build_vocab(["abc"])
    valid_data = tok.to_dict()

    # 1. Non-dict container
    with pytest.raises(TypeError, match="data must be a dict"):
        CharTokenizer.from_dict("not_a_dict")

    # 2. Missing container keys
    corrupt_keys = copy.deepcopy(valid_data)
    del corrupt_keys["format"]
    with pytest.raises(ValueError, match="Invalid tokenizer container keys"):
        CharTokenizer.from_dict(corrupt_keys)

    # 3. Format mismatch
    bad_format = copy.deepcopy(valid_data)
    bad_format["format"] = "unsupported-format"
    with pytest.raises(ValueError, match="Unsupported tokenizer format"):
        CharTokenizer.from_dict(bad_format)

    # 4. Version mismatch
    bad_version = copy.deepcopy(valid_data)
    bad_version["version"] = "99.0"
    with pytest.raises(ValueError, match="Unsupported tokenizer schema version"):
        CharTokenizer.from_dict(bad_version)

    # 5. Class type mismatch
    type_mismatch = copy.deepcopy(valid_data)
    type_mismatch["tokenizer_type"] = "WordTokenizer"
    with pytest.raises(ValueError, match="Tokenizer type mismatch"):
        CharTokenizer.from_dict(type_mismatch)

    # 6. Non-string token keys in vocab
    bad_key = copy.deepcopy(valid_data)
    bad_key["vocab"][123] = 4
    with pytest.raises(TypeError, match="All vocabulary token keys must be str"):
        CharTokenizer.from_dict(bad_key)

    # 7. Boolean ID in vocab
    bad_bool_id = copy.deepcopy(valid_data)
    bad_bool_id["vocab"]["a"] = True
    with pytest.raises(TypeError, match="Vocabulary ID for token 'a' must be an int"):
        CharTokenizer.from_dict(bad_bool_id)

    # 8. Negative ID in vocab
    bad_neg_id = copy.deepcopy(valid_data)
    bad_neg_id["vocab"]["a"] = -5
    with pytest.raises(ValueError, match="must be >= 0"):
        CharTokenizer.from_dict(bad_neg_id)

    # 9. Duplicate IDs in vocab
    bad_dup_id = copy.deepcopy(valid_data)
    bad_dup_id["vocab"]["b"] = bad_dup_id["vocab"]["a"]
    with pytest.raises(ValueError, match="Duplicate vocabulary ID"):
        CharTokenizer.from_dict(bad_dup_id)

    # 10. Non-contiguous IDs (gap in IDs)
    bad_gap_id = copy.deepcopy(valid_data)
    bad_gap_id["vocab"]["a"] = 999
    with pytest.raises(ValueError, match="must be strictly contiguous"):
        CharTokenizer.from_dict(bad_gap_id)

    # 11. Corrupted special token ID mapping
    bad_special = copy.deepcopy(valid_data)
    # swap PAD and UNK IDs
    bad_special["vocab"]["<PAD>"] = 1
    bad_special["vocab"]["<UNK>"] = 0
    with pytest.raises(ValueError, match="Special token '<PAD>' must have ID 0"):
        CharTokenizer.from_dict(bad_special)


def test_word_tokenizer_config_validation():
    tok = WordTokenizer().build_vocab(["hello world"])
    data = tok.to_dict()

    assert data["config"]["lexer"] == "unicode_word_whitespace_punctuation"
    assert data["config"]["lexer_version"] == "1"

    # Corrupt lexer config
    bad_config = copy.deepcopy(data)
    bad_config["config"]["lexer"] = "unsupported_lexer"
    with pytest.raises(ValueError, match="Unsupported lexer"):
        WordTokenizer.from_dict(bad_config)


def test_byte_tokenizer_config_validation():
    tok = ByteTokenizer()
    data = tok.to_dict()

    assert data["config"]["byte_offset"] == 4
    assert data["config"]["num_bytes"] == 256

    # Corrupt byte_offset
    bad_config = copy.deepcopy(data)
    bad_config["config"]["byte_offset"] = 0
    with pytest.raises(ValueError, match="Invalid byte_offset"):
        ByteTokenizer.from_dict(bad_config)


# =============================================================================
# Section 3: Unified Special Token Decode Semantics (P0-2)
# =============================================================================

def test_unified_special_token_decode_semantics_across_tokenizers():
    text = "hello"

    # 1. CharTokenizer
    char_tok = CharTokenizer().build_vocab([text])
    char_ids = char_tok.encode(text, add_bos=True, add_eos=True)
    assert char_tok.decode(char_ids, skip_special_tokens=True) == "hello"
    assert char_tok.decode(char_ids, skip_special_tokens=False) == "<BOS>hello<EOS>"

    # 2. ByteTokenizer
    byte_tok = ByteTokenizer()
    byte_ids = byte_tok.encode(text, add_bos=True, add_eos=True)
    assert byte_tok.decode(byte_ids, skip_special_tokens=True) == "hello"
    assert byte_tok.decode(byte_ids, skip_special_tokens=False) == "<BOS>hello<EOS>"

    # 3. WordTokenizer
    word_tok = WordTokenizer().build_vocab([text])
    word_ids = word_tok.encode(text, add_bos=True, add_eos=True)
    assert word_tok.decode(word_ids, skip_special_tokens=True) == "hello"
    assert word_tok.decode(word_ids, skip_special_tokens=False) == "<BOS>hello<EOS>"


# =============================================================================
# Section 4: Pure Python Isolated Subprocess Verification (P1-1)
# =============================================================================

def test_tokenization_pure_python_isolated_subprocess():
    code = """
import sys
sys.modules["numpy"] = None
sys.modules["torch"] = None

from termux_train.tokenization import CharTokenizer, ByteTokenizer, WordTokenizer

# CharTokenizer test
c_tok = CharTokenizer().build_vocab(["hello world 123"])
c_enc = c_tok.encode("hello", add_bos=True, add_eos=True)
assert c_tok.decode(c_enc, skip_special_tokens=True) == "hello"
assert c_tok.decode(c_enc, skip_special_tokens=False) == "<BOS>hello<EOS>"

# ByteTokenizer test
b_tok = ByteTokenizer()
b_enc = b_tok.encode("안녕하세요 🚀", add_bos=True, add_eos=True)
assert b_tok.decode(b_enc, skip_special_tokens=True) == "안녕하세요 🚀"

# WordTokenizer test
w_tok = WordTokenizer().build_vocab(["hello, world! 123"])
w_enc = w_tok.encode("hello, world!", add_bos=True, add_eos=True)
assert w_tok.decode(w_enc, skip_special_tokens=True) == "hello, world!"

print("SUBPROCESS_ISOLATION_PASS")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "SUBPROCESS_ISOLATION_PASS" in res.stdout


# =============================================================================
# Section 5: ByteTokenizer Argument & Edge Case Validations (P1-2)
# =============================================================================

def test_byte_tokenizer_strict_build_vocab_arguments():
    tok = ByteTokenizer()

    # texts type validation
    with pytest.raises(TypeError, match="texts must be a list or tuple"):
        tok.build_vocab(texts=123)

    with pytest.raises(TypeError, match="All elements in texts must be strings"):
        tok.build_vocab(texts=[123])

    # min_freq must be 1
    with pytest.raises(ValueError, match="min_freq must be 1"):
        tok.build_vocab(min_freq=2)

    # max_vocab_size must be None or 260
    with pytest.raises(ValueError, match="max_vocab_size must be None or 260"):
        tok.build_vocab(max_vocab_size=100)


def test_byte_tokenizer_strict_decode_error_handling():
    tok = ByteTokenizer()
    # Create invalid UTF-8 sequence (e.g. solitary continuation byte 0x80)
    invalid_byte_id = 0x80 + ByteTokenizer.BYTE_OFFSET

    with pytest.raises(UnicodeDecodeError):
        tok.decode([invalid_byte_id], errors="strict")

    # errors="replace" produces replacement character '\ufffd'
    res_replaced = tok.decode([invalid_byte_id], errors="replace")
    assert "\ufffd" in res_replaced


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


# =============================================================================
# Section 6: Empty Corpus Validation (P1-3)
# =============================================================================

def test_empty_corpus_rejections():
    # CharTokenizer
    with pytest.raises(ValueError, match="Cannot build vocabulary from an empty texts sequence"):
        CharTokenizer().build_vocab([])

    with pytest.raises(ValueError, match="Cannot build vocabulary from texts containing zero characters"):
        CharTokenizer().build_vocab(["", ""])

    # WordTokenizer
    with pytest.raises(ValueError, match="Cannot build vocabulary from an empty texts sequence"):
        WordTokenizer().build_vocab([])

    with pytest.raises(ValueError, match="Cannot build vocabulary from texts containing zero tokens"):
        WordTokenizer().build_vocab(["", ""])


# =============================================================================
# Section 7: CharTokenizer & WordTokenizer Determinism and Round-Trip
# =============================================================================

def test_char_tokenizer_deterministic_vocab_and_tie_breaking():
    texts = ["b b b a a c c"]
    tok1 = CharTokenizer().build_vocab(texts)
    tok2 = CharTokenizer().build_vocab(texts)

    assert tok1.get_vocab() == tok2.get_vocab()
    assert tok1.token_to_id("b") < tok1.token_to_id("a")
    assert tok1.token_to_id("a") < tok1.token_to_id("c")


def test_char_tokenizer_min_freq_and_max_vocab_size():
    corpus = ["apple banana cherry date elderberry"]
    tok = CharTokenizer().build_vocab(corpus, min_freq=3)
    assert tok.token_to_id("e") != UNK_ID
    assert tok.token_to_id("a") != UNK_ID
    assert tok.token_to_id("z") == UNK_ID

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

    assert encoded[0] == tok.token_to_id("a")
    assert encoded[1] == UNK_ID
    assert encoded[2] == UNK_ID

    decoded = tok.decode(encoded, skip_special_tokens=False)
    assert "<UNK>" in decoded


def test_char_tokenizer_unicode_multilingual_and_emoji():
    multilingual = "안녕하세요! 🚀 딥러닝 PyTorch Termux-Train (株) Café 123"
    tok = CharTokenizer().build_vocab([multilingual])

    encoded = tok.encode(multilingual)
    decoded = tok.decode(encoded)

    assert decoded == multilingual
    assert UNK_ID not in encoded


def test_word_tokenizer_whitespace_and_punctuation_preservation():
    text = "  Hello,   world! \n\tThis is a   DocFold <DOC> test.\n"
    tok = WordTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert decoded == text
    assert UNK_ID not in encoded


def test_word_tokenizer_deterministic_vocab_and_min_freq():
    corpus = ["apple banana apple cherry apple banana date"]
    tok = WordTokenizer().build_vocab(corpus, min_freq=2)

    assert tok.token_to_id("apple") < tok.token_to_id("banana")
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

    assert encoded[0] == tok.token_to_id("The")
    assert tok.token_to_id("lazy") == UNK_ID
    assert tok.token_to_id("dog") == UNK_ID


def test_all_tokenizers_empty_string():
    for TokClass in [CharTokenizer, ByteTokenizer, WordTokenizer]:
        tok = TokClass()
        if not tok.is_built:
            tok.build_vocab(["abc"])

        assert tok.encode("") == []
        assert tok.decode([]) == ""

        assert tok.encode("", add_bos=True, add_eos=True) == [BOS_ID, EOS_ID]
        assert tok.decode([BOS_ID, EOS_ID], skip_special_tokens=True) == ""


def test_special_token_literal_text_handling():
    text = "Value is <UNK> and <PAD>"
    tok = CharTokenizer().build_vocab([text])

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text
