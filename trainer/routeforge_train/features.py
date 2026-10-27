"""Shared feature extraction (spec 8.1).

These definitions are a byte-for-byte contract with the Go gateway
(`internal/policy/features.go`), pinned by `testdata/parity/features.jsonl`.
Do not change them without changing both sides.
"""

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

HASH_DIM = 262144
DENSE_NAMES = (
    "log_bytes",
    "log_tokens",
    "digit_frac",
    "upper_frac",
    "code_fences",
    "lines",
    "question_marks",
    "max_token_len",
)
VOCAB_MIN_DF = 5
VOCAB_MAX = 20000

_LOWER_TABLE = bytes(b + 32 if 65 <= b <= 90 else b for b in range(256))
_TOKEN_RE = re.compile(rb"[a-z0-9]+")
_DIGITS = b"0123456789"
_UPPER = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                raise ValueError("content part is not a text part")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError("text part has no string text")
            parts.append(text)
        return "\n".join(parts)
    raise ValueError("message content is missing or not text")


def feature_text(messages: Sequence[Mapping[str, object]]) -> str:
    """Join message contents in order with newlines; roles are ignored.

    Lone surrogates become U+FFFD, matching Go's JSON decoder.
    """
    text = "\n".join(_content_text(message.get("content")) for message in messages)
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(data: bytes) -> list[bytes]:
    """Lowercase ASCII, then split into maximal runs of [a-z0-9] bytes."""
    return _TOKEN_RE.findall(data.translate(_LOWER_TABLE))


def fnv1a32(data: bytes) -> int:
    h = 2166136261
    for byte in data:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def hashed_features(text: str) -> tuple[list[int], list[float]]:
    """Ascending hashed indices and their L2-normalized `1 + ln(count)` values."""
    tokens = tokenize(text.encode("utf-8"))
    counts: dict[int, int] = {}
    grams = list(tokens)
    grams.extend(tokens[i] + b" " + tokens[i + 1] for i in range(len(tokens) - 1))
    for gram in grams:
        index = fnv1a32(gram) & (HASH_DIM - 1)
        counts[index] = counts.get(index, 0) + 1
    indices = sorted(counts)
    values = [1.0 + math.log(counts[i]) for i in indices]
    squares = 0.0
    for value in values:
        squares += value * value
    norm = math.sqrt(squares)
    return indices, [value / norm for value in values]


def dense_raw(text: str) -> list[float]:
    """The eight raw dense features, in `DENSE_NAMES` order."""
    b = text.encode("utf-8")
    n = len(b)
    tokens = tokenize(b)
    digits = n - len(b.translate(None, _DIGITS))
    upper = n - len(b.translate(None, _UPPER))
    return [
        math.log(1 + n),
        math.log(1 + len(tokens)),
        digits / n if n else 0.0,
        upper / n if n else 0.0,
        float(b.count(b"```")),
        float(b.count(b"\n") + 1) if n else 0.0,
        float(b.count(b"?")),
        float(max((len(t) for t in tokens), default=0)),
    ]


@dataclass(frozen=True)
class DenseScaler:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(cls, raw: np.ndarray) -> DenseScaler:
        """Train mean and population std (ddof=0); a std of exactly 0.0 becomes 1.0."""
        if raw.shape[0] == 0:
            raise ValueError("cannot fit a scaler on zero rows")
        mean = raw.mean(axis=0)
        std = raw.std(axis=0, ddof=0)
        std = np.where(std == 0.0, 1.0, std)
        return cls(mean=tuple(float(v) for v in mean), std=tuple(float(v) for v in std))

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return (raw - np.array(self.mean)) / np.array(self.std)


@dataclass(frozen=True)
class Features:
    hashed: sparse.csr_matrix
    dense_raw: np.ndarray


def featurize(texts: Sequence[str]) -> Features:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    dense_rows: list[list[float]] = []
    for text in texts:
        idx, val = hashed_features(text)
        indices.extend(idx)
        data.extend(val)
        indptr.append(len(indices))
        dense_rows.append(dense_raw(text))
    hashed = sparse.csr_matrix(
        (
            np.array(data, dtype=np.float64),
            np.array(indices, dtype=np.int64),
            np.array(indptr, dtype=np.int64),
        ),
        shape=(len(texts), HASH_DIM),
    )
    dense = np.array(dense_rows, dtype=np.float64).reshape(len(texts), len(DENSE_NAMES))
    return Features(hashed=hashed, dense_raw=dense)


def select_vocab(
    hashed: sparse.csr_matrix, min_df: int = VOCAB_MIN_DF, max_size: int = VOCAB_MAX
) -> np.ndarray:
    """Hashed indices with document frequency >= min_df, most frequent first.

    Ties go to the smaller index; the result is truncated to max_size and sorted ascending.
    """
    df = np.asarray((hashed != 0).sum(axis=0)).ravel()
    candidates = np.flatnonzero(df >= min_df)
    order = np.lexsort((candidates, -df[candidates]))
    return np.sort(candidates[order][:max_size]).astype(np.int64)


def design_matrix(
    hashed: sparse.csr_matrix, dense_z: np.ndarray, vocab: np.ndarray
) -> sparse.csr_matrix:
    """Columns `[hashed[:, vocab] | dense_z]`."""
    return sparse.hstack(
        [hashed[:, vocab], sparse.csr_matrix(dense_z)], format="csr", dtype=np.float64
    )


def global_feature(column: int, vocab: np.ndarray) -> int:
    """Map a compact design-matrix column to its global feature id."""
    if column < len(vocab):
        return int(vocab[column])
    return HASH_DIM + column - len(vocab)
