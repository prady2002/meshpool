"""Tokenizer built from a GGUF checkpoint's own vocabulary.

A model's tokenizer is part of the model. Pairing real weights with an
approximate tokenizer produces text that is subtly wrong in a way that looks
like a quality problem rather than a bug, so this implements llama.cpp's actual
SentencePiece merge algorithm rather than a convenient approximation.

The algorithm (``llm_tokenizer_spm`` in llama.cpp) is:

1. split the text into UTF-8 symbols in a doubly linked list;
2. for every adjacent pair, if the concatenation is a vocabulary entry, push it
   onto a priority queue keyed by (score descending, position ascending);
3. pop the best pair, merge it if both halves are still intact and unchanged in
   size, then offer the two new neighbouring pairs to the queue;
4. emit ids for the surviving symbols, falling back to per-byte tokens for
   anything not in the vocabulary.

Byte-level BPE vocabularies (``tokenizer.ggml.model == "gpt2"``) use a different
pre-tokenization regex and merge table; rather than approximate them, they are
refused with a specific error. Guessing there would be the same silent-corruption
failure this module exists to avoid.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

_SPACE = "\u2581"          # U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's space


class TokenizerUnsupported(RuntimeError):
    """This checkpoint's tokenizer cannot be reproduced faithfully."""


@dataclass
class _Symbol:
    text: str
    prev: int
    nxt: int
    alive: bool = True


@dataclass
class GGUFTokenizer:
    """SentencePiece tokenizer reconstructed from GGUF metadata."""

    tokens: list[str]
    scores: list[float]
    token_types: list[int]
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    add_bos: bool = True
    # Both default to llama.cpp's behaviour, not sentencepiece's. The original
    # SentencePiece normalizer collapses whitespace runs and strips the ends;
    # llama.cpp deliberately does not, to stay byte-identical with the HF
    # tokenizers most checkpoints were actually trained and evaluated with.
    # GGUF records which one applies, so it is read from the file, never guessed.
    remove_extra_whitespaces: bool = False
    add_space_prefix: bool = True
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {t: i for i, t in enumerate(self.tokens)}
        if not self.scores:
            self.scores = [0.0] * len(self.tokens)

    @classmethod
    def from_vocab(cls, vocab: dict) -> "GGUFTokenizer":
        model = (vocab.get("model") or "").lower()
        if model not in ("llama", "spm", ""):
            raise TokenizerUnsupported(
                f"GGUF tokenizer model {model!r} is not SentencePiece. MeshPool "
                f"implements llama.cpp's SPM merge algorithm exactly; producing "
                f"approximate ids for a {model!r} vocabulary would corrupt output "
                f"in a way that looks like poor model quality. Convert the "
                f"checkpoint with an SPM tokenizer, or extend "
                f"meshpool.inference.gguf_tokenizer with the byte-level BPE path.")
        toks = list(vocab.get("tokens") or [])
        if not toks:
            raise TokenizerUnsupported("GGUF file carries no tokenizer vocabulary")
        return cls(
            tokens=toks,
            scores=[float(s) for s in (vocab.get("scores") or [])],
            token_types=[int(t) for t in (vocab.get("token_type") or [])],
            bos_id=vocab.get("bos_id"),
            eos_id=vocab.get("eos_id"),
            add_bos=bool(vocab.get("add_bos", True)),
            remove_extra_whitespaces=bool(vocab.get("remove_extra_whitespaces", False)),
            add_space_prefix=bool(vocab.get("add_space_prefix", True)),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _id(self, piece: str) -> Optional[int]:
        return self._index.get(piece)

    def _byte_ids(self, text: str) -> list[int]:
        """Per-byte fallback, matching llama.cpp's ``<0xAB>`` byte tokens."""
        out: list[int] = []
        for b in text.encode("utf-8"):
            tid = self._id(f"<0x{b:02X}>")
            if tid is None:
                unk = self._id("<unk>")
                if unk is None:
                    raise TokenizerUnsupported(
                        f"byte 0x{b:02X} has no token and the vocabulary has no "
                        f"<unk>; this text cannot be represented")
                out.append(unk)
            else:
                out.append(tid)
        return out

    def encode(self, text: str, add_bos: Optional[bool] = None) -> list[int]:
        """Text -> token ids, using llama.cpp's SPM bigram-merge order."""
        want_bos = self.add_bos if add_bos is None else add_bos
        ids: list[int] = []
        if want_bos and self.bos_id is not None:
            ids.append(int(self.bos_id))
        if self.remove_extra_whitespaces:
            text = " ".join(text.split())
        if not text:
            return ids

        # llama.cpp prepends a space and represents spaces with U+2581.
        work = (" " + text) if self.add_space_prefix else text
        work = work.replace(" ", _SPACE)

        syms: list[_Symbol] = []
        for ch in work:
            syms.append(_Symbol(ch, len(syms) - 1, len(syms) + 1))
        if syms:
            syms[-1].nxt = -1

        heap: list[tuple[float, int, int, int, int]] = []

        def offer(left: int, right: int) -> None:
            if left < 0 or right < 0 or right >= len(syms):
                return
            merged = syms[left].text + syms[right].text
            tid = self._id(merged)
            if tid is None:
                return
            # Negated score so heapq (a min-heap) pops the highest score first;
            # ties break on position, which is what llama.cpp does.
            heapq.heappush(heap, (-self.scores[tid], left, right, len(merged), tid))

        for i in range(len(syms) - 1):
            offer(i, i + 1)

        while heap:
            _neg, left, right, size, _tid = heapq.heappop(heap)
            ls, rs = syms[left], syms[right]
            if not (ls.alive and rs.alive):
                continue
            if len(ls.text) + len(rs.text) != size:
                continue                       # one side has since grown; stale
            ls.text += rs.text
            rs.alive = False
            ls.nxt = rs.nxt
            if rs.nxt != -1:
                syms[rs.nxt].prev = left
            offer(ls.prev, left)
            offer(left, ls.nxt)

        for s in syms:
            if not s.alive:
                continue
            tid = self._id(s.text)
            ids.extend([tid] if tid is not None else self._byte_ids(s.text))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text, undoing the space marker and byte fallback."""
        out: list[str] = []
        pending = bytearray()

        def flush() -> None:
            if pending:
                out.append(pending.decode("utf-8", errors="replace"))
                pending.clear()

        for tid in ids:
            if not (0 <= int(tid) < len(self.tokens)):
                continue
            piece = self.tokens[int(tid)]
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    pending.append(int(piece[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush()
            if self.token_types and int(tid) < len(self.token_types):
                # Control tokens (BOS/EOS/pad) carry no text.
                if self.token_types[int(tid)] == 3:
                    continue
            out.append(piece)
        flush()
        return "".join(out).replace(_SPACE, " ").lstrip(" ") if out else ""
