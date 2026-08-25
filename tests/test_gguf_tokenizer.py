"""The tokenizer is checked against SentencePiece itself, not against itself.

A tokenizer test that only round-trips ``decode(encode(x)) == x`` proves almost
nothing: an implementation can be self-consistently wrong and pass every such
test while producing different token ids than the model was trained on. Wrong
ids do not crash -- they degrade output quality in a way that is indistinguishable
from "the model is just bad", which is the worst kind of bug to own.

So the reference here is the real ``sentencepiece`` library. A vocabulary is
trained with it, that exact vocabulary is handed to ``GGUFTokenizer``, and the
two encoders must agree token-for-token. That is a genuinely independent oracle:
nothing in MeshPool influences what SentencePiece returns.
"""
from __future__ import annotations

import contextlib
import os
import random
import string
import unicodedata

import pytest

from meshpool.inference.gguf_tokenizer import GGUFTokenizer, TokenizerUnsupported

spm = pytest.importorskip("sentencepiece", reason="reference tokenizer not installed")

CORPUS_LINES = [
    "the quick brown fox jumps over the lazy dog",
    "meshpool runs large language models on pooled consumer hardware",
    "decentralized inference pipelines amortize latency across tokens",
    "speculative decoding verifies many tokens in one pipeline traversal",
    "constellations minimise pipeline depth times round trip time",
    "erasure coding survives the loss of most of its pieces",
    "a node that goes offline is replaced by a warm spare",
]


@pytest.fixture(scope="module")
def sp_model(tmp_path_factory):
    """Train a real SentencePiece BPE vocabulary to act as the oracle.

    BPE, not unigram: llama.cpp's SPM tokenizer merges the highest-scoring
    adjacent pair repeatedly, which is BPE's algorithm. Unigram uses Viterbi
    over the whole segmentation and would not be the same function.
    """
    d = tmp_path_factory.mktemp("spm")
    corpus = d / "corpus.txt"
    corpus.write_text("\n".join(CORPUS_LINES * 60), encoding="utf-8")
    prefix = str(d / "vocab")
    # The trainer logs to C-level stderr; keep the test output readable.
    with open(os.devnull, "w") as dn, contextlib.redirect_stderr(dn):
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=prefix, vocab_size=400,
            model_type="bpe", byte_fallback=True, character_coverage=1.0,
            bos_id=1, eos_id=2, unk_id=0, pad_id=-1)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


@pytest.fixture(scope="module")
def vocab(sp_model):
    """The oracle's own vocabulary, in the shape ``read_vocab`` produces."""
    n = sp_model.get_piece_size()
    return {
        "model": "llama",
        "tokens": [sp_model.id_to_piece(i) for i in range(n)],
        "scores": [sp_model.get_score(i) for i in range(n)],
        "token_type": [3 if sp_model.is_control(i) else 1 for i in range(n)],
        "bos_id": sp_model.bos_id(),
        "eos_id": sp_model.eos_id(),
        "add_bos": False,
        # SentencePiece's default normalizer collapses whitespace runs and
        # strips the ends. Matching the oracle means opting into that here;
        # real llama.cpp checkpoints ship this flag false, which is the default.
        "remove_extra_whitespaces": True,
    }


@pytest.fixture(scope="module")
def tok(vocab):
    return GGUFTokenizer.from_vocab(vocab)


CASES = [
    "hello world",
    "the quick brown fox",
    "meshpool",
    "",
    "decentralized inference",
    "xyzzy qwerty",
    "a",
    "  spaced  out  ",
    "Kimi K2 has 1T params",
    "unicode: caf\u00e9 na\u00efve",
    "\u4f60\u597d\u4e16\u754c",
    "the lazy dog jumps",
    "speculative decoding verifies many tokens",
    "1234567890",
    "MiXeD CaSe TeXt",
    "constellations minimise pipeline depth times round trip time",
    "erasure coding survives",
    "punctuation! and? symbols: @#$%^&*()",
    "\n\t whitespace \r\n mix \t",
    "a node that goes offline is replaced by a warm spare",
]


class TestMatchesTheReferenceImplementation:
    @pytest.mark.parametrize("text", CASES)
    def test_encode_is_identical_to_sentencepiece(self, tok, sp_model, text):
        assert tok.encode(text, add_bos=False) == sp_model.encode(text, out_type=int), (
            f"token ids diverge from the reference for {text!r}; "
            f"mine={[tok.tokens[i] for i in tok.encode(text, add_bos=False)]} "
            f"ref={[sp_model.id_to_piece(i) for i in sp_model.encode(text, out_type=int)]}")

    def test_encode_matches_on_random_text_too(self, tok, sp_model):
        """Hand-picked cases encode the author's blind spots; random text does not."""
        rng = random.Random(20240517)
        alphabet = string.ascii_letters + string.digits + " " * 6 + ".,!?-'"
        mismatches = []
        for _ in range(400):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
            if tok.encode(text, add_bos=False) != sp_model.encode(text, out_type=int):
                mismatches.append(text)
        assert not mismatches, f"{len(mismatches)}/400 random strings diverge, e.g. {mismatches[:3]!r}"

    def test_encode_matches_on_random_unicode(self, tok, sp_model):
        """Byte-fallback is where naive implementations quietly disagree.

        Restricted to characters SentencePiece's normalizer leaves alone. Its
        default ``nmt_nfkc`` normalizer both applies NFKC *and* deletes control
        characters -- ``sp.encode("\\x7f")`` returns ``[]``. llama.cpp does
        neither (see ``TestNormalizationIsDeliberatelyLlamaCppsNotSentencePieces``).
        Feeding either kind of character here would measure that known,
        intentional difference instead of the merge algorithm this test is about.
        """
        rng = random.Random(99)
        mismatches = []
        checked = 0
        while checked < 200:
            text = "".join(chr(rng.randint(32, 0x2FF)) for _ in range(rng.randint(1, 24)))
            if unicodedata.normalize("NFKC", text) != text:
                continue
            if any(unicodedata.category(c) in ("Cc", "Cf") for c in text):
                continue
            checked += 1
            if tok.encode(text, add_bos=False) != sp_model.encode(text, out_type=int):
                mismatches.append(text)
        assert not mismatches, f"{len(mismatches)}/200 unicode strings diverge, e.g. {mismatches[:2]!r}"

    def test_multibyte_characters_use_byte_fallback(self, tok, sp_model):
        """Non-vacuity guard for the test above: it must exercise byte tokens."""
        ids = tok.encode("\u00d7\u010b\u0127", add_bos=False)
        pieces = [tok.tokens[i] for i in ids]
        assert any(p.startswith("<0x") for p in pieces), (
            f"no byte-fallback tokens produced ({pieces}); the unicode test "
            f"would be passing without testing the fallback path")
        assert ids == sp_model.encode("\u00d7\u010b\u0127", out_type=int)

    def test_the_comparison_is_not_vacuous(self, tok, sp_model):
        """Guard against both sides silently returning nothing."""
        ids = tok.encode("the quick brown fox jumps", add_bos=False)
        assert len(ids) >= 3, f"suspiciously few tokens: {ids}"
        assert ids == sp_model.encode("the quick brown fox jumps", out_type=int)
        assert len(set(ids)) > 1, "all tokens identical -- encoder is degenerate"


class TestDecode:
    @pytest.mark.parametrize("text", [c for c in CASES if c.strip()])
    def test_round_trip_recovers_the_text(self, tok, text):
        normalized = " ".join(text.split())
        assert tok.decode(tok.encode(text, add_bos=False)) == normalized

    def test_decode_matches_the_reference(self, tok, sp_model):
        for text in CASES:
            ids = sp_model.encode(text, out_type=int)
            assert tok.decode(ids) == sp_model.decode(ids), f"decode diverges for {text!r}"

    def test_bos_is_not_rendered_as_text(self, tok):
        with_bos = tok.encode("hello world", add_bos=True)
        assert with_bos[0] == tok.bos_id
        assert tok.decode(with_bos) == "hello world"

    def test_out_of_range_ids_are_skipped_not_crashed(self, tok):
        assert tok.decode([999999, -1]) == ""


class TestBosHandling:
    def test_add_bos_defaults_to_the_checkpoint_setting(self, vocab):
        assert GGUFTokenizer.from_vocab({**vocab, "add_bos": True}).encode("hi")[0] == 1
        assert GGUFTokenizer.from_vocab({**vocab, "add_bos": False}).encode("hi")[0] != 1

    def test_explicit_argument_overrides_the_checkpoint(self, vocab):
        t = GGUFTokenizer.from_vocab({**vocab, "add_bos": False})
        assert t.encode("hi", add_bos=True)[0] == 1

    def test_bos_is_added_even_to_empty_text(self, vocab):
        t = GGUFTokenizer.from_vocab({**vocab, "add_bos": True})
        assert t.encode("") == [1]


class TestNormalizationFlagsComeFromTheFile:
    def test_whitespace_collapsing_is_off_by_default(self, vocab):
        """llama.cpp's default. Turning it on silently would change every prompt."""
        v = dict(vocab)
        v.pop("remove_extra_whitespaces")
        assert GGUFTokenizer.from_vocab(v).remove_extra_whitespaces is False

    def test_collapsing_actually_changes_the_ids(self, vocab):
        on = GGUFTokenizer.from_vocab({**vocab, "remove_extra_whitespaces": True})
        off = GGUFTokenizer.from_vocab({**vocab, "remove_extra_whitespaces": False})
        text = "  spaced  out  "
        assert on.encode(text) != off.encode(text), (
            "the flag has no effect, so honouring it proves nothing")

    def test_space_prefix_can_be_disabled(self, vocab):
        on = GGUFTokenizer.from_vocab({**vocab, "add_space_prefix": True})
        off = GGUFTokenizer.from_vocab({**vocab, "add_space_prefix": False})
        assert on.encode("hello") != off.encode("hello")


class TestNormalizationIsDeliberatelyLlamaCppsNotSentencePieces:
    """MeshPool matches llama.cpp, which does not Unicode-normalize. That is a
    real, intentional divergence from the ``sentencepiece`` library, and it is
    pinned here so a future change cannot quietly adopt the other convention.

    Real GGUF checkpoints are served by llama.cpp, so llama.cpp's ids are the
    ones the model actually saw. Adding NFKC here would make this test file's
    oracle happier and the product wrong.
    """

    def test_nfkc_unstable_input_diverges_from_sentencepiece(self, tok, sp_model):
        text = "\u0133"          # U+0133 LATIN SMALL LIGATURE IJ; NFKC -> "ij"
        assert unicodedata.normalize("NFKC", text) != text, "case is no longer NFKC-unstable"
        ref_pieces = sp_model.encode(text, out_type=str)
        assert "i" in ref_pieces and "j" in ref_pieces, (
            "sentencepiece no longer decomposes the ligature; the premise changed")
        mine = [tok.tokens[i] for i in tok.encode(text, add_bos=False)]
        assert "i" not in mine and "j" not in mine, (
            "MeshPool applied Unicode normalization. llama.cpp does not, so this "
            "would silently change the token ids for every real checkpoint.")
        assert any(p.startswith("<0x") for p in mine), (
            f"expected raw byte fallback for the un-normalized ligature, got {mine}")

    def test_control_characters_are_kept_not_dropped(self, tok, sp_model):
        """The other half of the divergence: nmt_nfkc deletes them, we keep them."""
        assert sp_model.encode("\x7f", out_type=int) == [], (
            "sentencepiece no longer deletes control characters; premise changed")
        mine = tok.encode("a\x7fb", add_bos=False)
        assert mine != sp_model.encode("a\x7fb", out_type=int)
        assert tok.decode(mine) == "a\x7fb", (
            "a control character in the prompt must survive the round trip")

    def test_the_divergence_does_not_leak_into_ascii(self, tok, sp_model):
        """The divergence must be confined to NFKC-unstable characters."""
        for text in ("hello world", "meshpool", "the quick brown fox"):
            assert tok.encode(text, add_bos=False) == sp_model.encode(text, out_type=int)


class TestFailsLoudly:
    def test_a_byte_level_bpe_vocabulary_is_refused(self, vocab):
        with pytest.raises(TokenizerUnsupported, match="not SentencePiece"):
            GGUFTokenizer.from_vocab({**vocab, "model": "gpt2"})

    def test_an_empty_vocabulary_is_refused(self):
        with pytest.raises(TokenizerUnsupported, match="no tokenizer vocabulary"):
            GGUFTokenizer.from_vocab({"model": "llama", "tokens": []})

    def test_the_refusal_names_the_offending_model(self, vocab):
        with pytest.raises(TokenizerUnsupported) as e:
            GGUFTokenizer.from_vocab({**vocab, "model": "rwkv"})
        assert "rwkv" in str(e.value), "error must say which tokenizer was rejected"
