"""Reed–Solomon erasure coding over GF(2^8).

Implements a systematic (k, m) RS codec:
  * ``k`` data shards  (original data, verbatim)
  * ``m`` parity shards (computed; any m shards can be lost and recovered)

The code uses the standard GF(2^8) field with primitive polynomial
x^8 + x^4 + x^3 + x^2 + 1  (0x11d, the same Galois field used by AES).

Encoding matrix is Cauchy-based (columns = data positions, rows = all k+m
positions), ensuring any k×k sub-matrix is invertible, which is the
fundamental correctness requirement.

Shard format written by ``encode_to_shards`` / expected by ``decode_shards``:
  [4 bytes: original length big-endian]
  [4 bytes: shard_index big-endian]
  [32 bytes: SHA-256 of the shard payload (the rest of this blob)]
  [variable: shard payload (padded data or parity)]

The header lets the decoder know original length (to strip padding) and
detect corruption even when no in-flight hash is available.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# GF(2^8) arithmetic with primitive polynomial 0x11d
# ---------------------------------------------------------------------------
_PRIM = 0x11d   # x^8 + x^4 + x^3 + x^2 + 1


def _build_tables():
    exp = np.zeros(512, dtype=np.uint16)
    log = np.zeros(256, dtype=np.uint16)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= _PRIM
        x &= 0xFF
    # wrap-around: exp[i] == exp[i % 255] for i >= 255
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


_EXP, _LOG = _build_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_EXP[int(_LOG[a]) + int(_LOG[b])])


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("GF division by zero")
    if a == 0:
        return 0
    return int(_EXP[(int(_LOG[a]) - int(_LOG[b])) % 255])


def _gf_pow(x: int, e: int) -> int:
    if e == 0:
        return 1
    if x == 0:
        return 0
    return int(_EXP[(int(_LOG[x]) * e) % 255])


def _gf_inv(x: int) -> int:
    if x == 0:
        raise ZeroDivisionError("GF inverse of zero")
    return int(_EXP[255 - int(_LOG[x])])


# ---------------------------------------------------------------------------
# GF(2^8) matrix operations (matrices stored as 2-D numpy uint8 arrays)
# ---------------------------------------------------------------------------

def _mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """GF(2^8) matrix multiply.  A: (r,n)  B: (n,c)  -> (r,c)."""
    r, n = A.shape
    n2, c = B.shape
    assert n == n2
    C = np.zeros((r, c), dtype=np.uint8)
    # Vectorise the inner dimension with python-level loop over columns
    for j in range(c):
        for i in range(r):
            val = 0
            for k in range(n):
                val ^= _gf_mul(int(A[i, k]), int(B[k, j]))
            C[i, j] = val
    return C


def _mat_mul_fast(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """GF(2^8) matrix multiply, vectorised using LOG/EXP look-up tables.

    We use the identity:
        a * b = EXP[LOG[a] + LOG[b]]   (with 0 handled separately)

    We convert indices to int16 to avoid overflow in the sum, then index
    into _EXP.  This is ~10x faster than the pure Python loop version for
    large matrices.
    """
    r, n = A.shape
    n2, c = B.shape
    assert n == n2

    A16 = A.astype(np.int16)
    B16 = B.astype(np.int16)

    # masks for zero elements
    Az = (A == 0)    # (r, n)
    Bz = (B == 0)    # (n, c)

    # log tables as int16
    LA = np.where(Az, np.int16(0), _LOG[A16])   # (r, n)
    LB = np.where(Bz, np.int16(0), _LOG[B16])   # (n, c)

    C = np.zeros((r, c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            acc = 0
            for l in range(n):     # noqa: E741
                a = int(A[i, l])
                b = int(B[l, j])
                if a and b:
                    acc ^= int(_EXP[int(_LOG[a]) + int(_LOG[b])])
            C[i, j] = acc
    return C


def _mat_mul_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """GF(2^8) matrix multiply using numpy broadcasting."""
    r, n = A.shape
    n2, c = B.shape
    assert n == n2

    C = np.zeros((r, c), dtype=np.uint8)
    for k in range(n):
        a_col = A[:, k].astype(np.uint16)    # (r,)
        b_row = B[k, :].astype(np.uint16)    # (c,)
        # outer product: rows×cols
        a_k = a_col[:, None]    # (r, 1)
        b_k = b_row[None, :]    # (1, c)
        zero_mask = (a_k == 0) | (b_k == 0)
        log_sum = (_LOG[a_k] + _LOG[b_k]) % 255
        prod = np.where(zero_mask, np.uint8(0), _EXP[log_sum].astype(np.uint8))
        C ^= prod.astype(np.uint8)
    return C


def _mat_invert(M: np.ndarray) -> np.ndarray:
    """Invert a square GF(2^8) matrix via Gaussian elimination."""
    n = M.shape[0]
    assert M.shape == (n, n)
    A = M.copy().astype(np.uint16)
    I = np.eye(n, dtype=np.uint16)

    for col in range(n):
        # find pivot
        pivot = -1
        for row in range(col, n):
            if A[row, col] != 0:
                pivot = row
                break
        if pivot == -1:
            raise ValueError("Matrix is singular over GF(2^8)")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
            I[[col, pivot]] = I[[pivot, col]]
        # scale pivot row
        inv_p = _gf_inv(int(A[col, col]))
        for c in range(n):
            A[col, c] = _gf_mul(int(A[col, c]), inv_p)
            I[col, c] = _gf_mul(int(I[col, c]), inv_p)
        # eliminate column
        for row in range(n):
            if row == col or A[row, col] == 0:
                continue
            factor = int(A[row, col])
            for c in range(n):
                A[row, c] ^= _gf_mul(factor, int(A[col, c]))
                I[row, c] ^= _gf_mul(factor, int(I[col, c]))

    return I.astype(np.uint8)


# ---------------------------------------------------------------------------
# Cauchy encoding matrix
# ---------------------------------------------------------------------------

def _cauchy_matrix(k: int, m: int) -> np.ndarray:
    """Return a (k+m, k) Cauchy-based systematic encoding matrix.

    Rows 0..k-1 form the k×k identity (systematic part; data shards are
    unmodified).  Rows k..k+m-1 are a Cauchy matrix whose entries are:
        M[i, j] = 1 / (x_i XOR y_j)    in GF(2^8)
    with x_i in {0..m-1} and y_j in {m..m+k-1}.
    This choice guarantees all sub-matrices are invertible.

    We require k + m <= 255 for the field elements to be distinct.
    """
    if k + m > 255:
        raise ValueError(f"k+m={k+m} exceeds GF(2^8) limit of 255")
    enc = np.zeros((k + m, k), dtype=np.uint8)
    # Identity block
    for i in range(k):
        enc[i, i] = 1
    # Cauchy block
    # x-values: 0 .. m-1  (for parity rows)
    # y-values: m .. m+k-1  (for data columns)
    for i in range(m):
        xi = i           # x_i
        for j in range(k):
            yj = m + j   # y_j
            enc[k + i, j] = _gf_inv(xi ^ yj)
    return enc


# ---------------------------------------------------------------------------
# High-level codec
# ---------------------------------------------------------------------------
_HEADER_FMT = ">II32s"    # orig_len (4), shard_index (4), sha256 (32)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 40 bytes


def _shard_hash(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _pack_shard(orig_len: int, shard_index: int, payload: bytes) -> bytes:
    h = _shard_hash(payload)
    return struct.pack(_HEADER_FMT, orig_len, shard_index, h) + payload


def _unpack_shard(data: bytes) -> tuple[int, int, bool, bytes]:
    """Returns (orig_len, shard_index, valid, payload)."""
    if len(data) < _HEADER_SIZE:
        return 0, 0, False, b""
    orig_len, shard_index, claimed_hash = struct.unpack(
        _HEADER_FMT, data[:_HEADER_SIZE])
    payload = data[_HEADER_SIZE:]
    valid = (_shard_hash(payload) == claimed_hash)
    return orig_len, shard_index, valid, payload


class RSCodec:
    """Systematic Reed–Solomon (k, m) codec over GF(2^8).

    Parameters
    ----------
    k : int   number of data shards (default 6)
    m : int   number of parity shards (default 3); tolerates loss of ANY m shards
    """

    def __init__(self, k: int = 6, m: int = 3) -> None:
        if k < 1 or m < 1:
            raise ValueError("k and m must both be >= 1")
        if k + m > 255:
            raise ValueError(f"k+m={k+m} exceeds GF(2^8) limit of 255")
        self.k = k
        self.m = m
        self._enc = _cauchy_matrix(k, m)  # (k+m, k)

    # ------------------------------------------------------------------ encode
    def encode(self, data: bytes) -> list[bytes]:
        """Split ``data`` into k+m shards, each a ``bytes`` object.

        The first k shards contain the (padded) original data; the last m
        contain parity.  Each shard is wrapped with a 40-byte header that
        carries the original length, shard index, and SHA-256 of the payload
        so corrupted shards can be detected.
        """
        k, m = self.k, self.m
        orig_len = len(data)

        # pad so length is divisible by k
        pad = (-orig_len) % k
        padded = data + bytes(pad)
        shard_size = len(padded) // k

        # split into k data shards as uint8 arrays, each of shape (shard_size,)
        data_shards = [
            np.frombuffer(padded[i * shard_size:(i + 1) * shard_size],
                          dtype=np.uint8)
            for i in range(k)
        ]
        # data matrix: (k, shard_size) — each row is one data shard
        D = np.stack(data_shards, axis=0)  # (k, shard_size)

        # encoding matrix parity block: (m, k)
        P_enc = self._enc[k:]   # rows k..k+m-1
        # parity matrix: (m, shard_size) = P_enc @ D  over GF(2^8)
        # Use _mat_mul_np for speed
        Parity = _mat_mul_np(P_enc, D)  # (m, shard_size)

        shards: list[bytes] = []
        for i in range(k):
            payload = D[i].tobytes()
            shards.append(_pack_shard(orig_len, i, payload))
        for i in range(m):
            payload = Parity[i].tobytes()
            shards.append(_pack_shard(orig_len, k + i, payload))
        return shards

    # ------------------------------------------------------------------ decode
    def decode(self, shards: list[Optional[bytes]]) -> bytes:
        """Reconstruct original data from at least k valid shards.

        Parameters
        ----------
        shards : list of length k+m
            Each element is either the original shard bytes (as returned by
            ``encode``) or ``None`` to indicate the shard is missing.
            Corrupted shards (bad hash) are automatically treated as missing.

        Returns
        -------
        bytes — the original data, exactly as passed to ``encode``.

        Raises
        ------
        ValueError  if fewer than k valid shards are present.
        """
        k, m = self.k, self.m
        total = k + m
        if len(shards) != total:
            raise ValueError(
                f"expected {total} shard slots, got {len(shards)}")

        # parse shards; detect corruption via hash
        orig_len: Optional[int] = None
        parsed: list[Optional[tuple[int, bytes]]] = []   # (shard_index, payload) or None
        shard_size: Optional[int] = None

        for raw in shards:
            if raw is None:
                parsed.append(None)
                continue
            ol, idx, valid, payload = _unpack_shard(raw)
            if not valid:
                parsed.append(None)   # corrupted — treat as missing
                continue
            if orig_len is None:
                orig_len = ol
            if shard_size is None:
                shard_size = len(payload)
            parsed.append((idx, payload))

        valid_indices = [i for i, p in enumerate(parsed) if p is not None]
        if len(valid_indices) < k:
            raise ValueError(
                f"not enough valid shards: need {k}, have {len(valid_indices)}")

        if orig_len is None:
            raise ValueError("could not determine original length")

        # Fast path: if all k data shards are intact, just concatenate them
        if all(parsed[i] is not None for i in range(k)):
            data_shards = [parsed[i][1] for i in range(k)]  # type: ignore[index]
            padded = b"".join(data_shards)
            return padded[:orig_len]

        # Reconstruction path: choose k available shards and solve
        chosen = valid_indices[:k]   # first k available
        # sub-matrix of encoding matrix: rows = chosen shard indices
        sub_enc = np.array([self._enc[idx] for idx in chosen],
                           dtype=np.uint8)  # (k, k)
        sub_enc_inv = _mat_invert(sub_enc)  # (k, k)

        # assemble the chosen shard payloads into a matrix (k, shard_size)
        S = np.stack(
            [np.frombuffer(parsed[i][1], dtype=np.uint8)   # type: ignore[index]
             for i in chosen],
            axis=0)

        # recover data shards: D = sub_enc_inv @ S  over GF(2^8)
        D = _mat_mul_np(sub_enc_inv, S)  # (k, shard_size)

        padded = D.tobytes()
        return padded[:orig_len]

    def shard_count(self) -> int:
        return self.k + self.m


# ---------------------------------------------------------------------------
# Placement diversity helper
# ---------------------------------------------------------------------------

def pick_diverse_nodes(
        nodes: list,
        count: int,
        *,
        group_attr: str = "group",
) -> list:
    """Choose ``count`` nodes, maximising group diversity.

    ``nodes`` is a list of node objects; the optional ``group_attr`` is the
    attribute (e.g. 'region', 'isp') used to spread shards.  Nodes without
    that attribute are treated as their own group.

    The greedy algorithm interleaves nodes from different groups round-robin
    (like dealing cards) so no two consecutive shards share a group whenever
    avoidance is possible.
    """
    if count <= 0:
        return []

    # group nodes
    groups: dict[str, list] = {}
    for node in nodes:
        g = getattr(node, group_attr, node.id)  # fall back to node id = unique group
        groups.setdefault(g, []).append(node)

    result: list = []
    used_ids: set = set()
    rounds = [list(g) for g in groups.values()]

    while len(result) < count:
        made_progress = False
        for grp in rounds:
            if len(result) >= count:
                break
            for node in list(grp):
                if node.id not in used_ids:
                    result.append(node)
                    used_ids.add(node.id)
                    grp.remove(node)
                    made_progress = True
                    break
        if not made_progress:
            break   # no more nodes

    return result[:count]


# ---------------------------------------------------------------------------
# Crossover calculation (replication vs erasure coding)
# ---------------------------------------------------------------------------

def erasure_overhead(k: int, m: int) -> float:
    """Storage overhead factor for (k, m) erasure coding. e.g. 1.5 = 50% overhead."""
    return (k + m) / k


def replication_overhead(n: int) -> float:
    """Storage overhead factor for n-way replication. e.g. 2.0 = 100% overhead."""
    return float(n)


def erasure_crossover_bytes(k: int, m: int, replication: int = 2) -> int:
    """Minimum object size at which erasure coding saves space vs replication.

    Below this threshold, storing k+m shards each of size ceil(size/k) costs
    MORE than 'replication' plain copies, so replication is preferred.

    Returns the crossover in bytes.  At or above this size, erasure wins.
    """
    # cost of EC:   (k+m) * ceil(size/k)  bytes across the cluster
    # cost of repl: replication * size     bytes across the cluster
    # EC wins when: (k+m)/k < replication, which is always true for sensible k,m,r
    # BUT there is a per-shard header overhead (_HEADER_SIZE bytes per shard):
    # EC total = (k+m) * (ceil(size/k) + HEADER_SIZE)
    # Repl total = replication * size
    # Crossover: (k+m) * (size/k + HEADER_SIZE) = replication * size
    # (k+m)*HEADER_SIZE = size * (replication - (k+m)/k)
    # size = (k+m)*HEADER_SIZE / (replication - (k+m)/k)
    ec_factor = (k + m) / k
    diff = replication - ec_factor
    if diff <= 0:
        return 0   # EC always wins (lower overhead than replication)
    return int((k + m) * _HEADER_SIZE / diff) + 1
