import math

import numpy as np
import pytest
from scipy import sparse

from routeforge_train.features import (
    DENSE_NAMES,
    HASH_DIM,
    VOCAB_MAX,
    VOCAB_MIN_DF,
    DenseScaler,
    dense_raw,
    design_matrix,
    feature_text,
    featurize,
    fnv1a32,
    global_feature,
    hashed_features,
    prompt_sha256,
    select_vocab,
    tokenize,
)


def test_constants():
    assert HASH_DIM == 262144
    assert DENSE_NAMES == (
        "log_bytes",
        "log_tokens",
        "digit_frac",
        "upper_frac",
        "code_fences",
        "lines",
        "question_marks",
        "max_token_len",
    )
    assert VOCAB_MIN_DF == 5
    assert VOCAB_MAX == 20000


def test_fnv1a32_known_vectors():
    assert fnv1a32(b"") == 0x811C9DC5
    assert fnv1a32(b"a") == 0xE40C292C
    assert fnv1a32(b"foobar") == 0xBF9CF968


def test_tokenize_lowercases_ascii_and_splits_on_other_bytes():
    assert tokenize(b"Hello, World! x2") == [b"hello", b"world", b"x2"]
    assert tokenize("Café ÀB".encode()) == [b"caf", b"b"]
    assert tokenize(b"") == []
    assert tokenize(b"?!") == []


def test_feature_text_joins_messages_and_text_parts():
    msgs = [
        {"role": "system", "content": "a"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "b"}, {"type": "text", "text": "c"}],
        },
        {"role": "assistant", "content": "d"},
    ]
    assert feature_text(msgs) == "a\nb\nc\nd"


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "image_url"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}]}],
        [{"role": "user", "content": None}],
        [{"role": "user"}],
    ],
)
def test_feature_text_rejects_non_text_parts_and_null_content(messages):
    with pytest.raises(ValueError):
        feature_text(messages)


def test_feature_text_replaces_lone_surrogates():
    assert feature_text([{"role": "user", "content": "a\ud800b"}]) == "a\ufffdb"
    assert feature_text([{"role": "user", "content": "\U0001f680"}]) == "\U0001f680"


def test_hashed_features_known_answer():
    idx, val = hashed_features("The the cat")
    assert idx == [23559, 23896, 176668, 217319]  # cat, "the cat", the, "the the"
    a, b = 0.6990303272568005, 0.4128585720620119  # (1+ln2)/norm, 1/norm
    assert val == pytest.approx([b, b, a, b], abs=1e-15)


def test_hashed_features_aggregates_collisions_by_index():
    # b"t368" and b"t3226" both hash to index 91390
    idx, val = hashed_features("t368 t3226")
    c = 1 + math.log(2)
    norm = math.sqrt(c * c + 1)
    assert len(idx) == 2
    assert dict(zip(idx, val, strict=True))[91390] == pytest.approx(c / norm, abs=1e-15)


def test_hashed_features_empty_and_separator_only():
    assert hashed_features("") == ([], [])
    assert hashed_features("?! ...") == ([], [])


def test_hashed_values_unit_norm():
    idx, val = hashed_features("Write a Python function")
    assert idx == sorted(idx)
    assert sum(v * v for v in val) == pytest.approx(1.0, abs=1e-15)


def test_dense_raw_known_answer():
    assert dense_raw("Hi?\n```x```\nABC 12") == pytest.approx(
        [math.log(19), math.log(5), 2 / 18, 4 / 18, 2.0, 3.0, 1.0, 3.0], abs=1e-15
    )
    assert dense_raw("") == [0.0] * 8


def test_dense_raw_counts_lines_by_newline_byte():
    assert dense_raw("a\r\nb\r\n")[5] == 3.0
    assert dense_raw("````")[4] == 1.0


def test_dense_scaler_zero_std_becomes_one():
    s = DenseScaler.fit(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert s.mean == (2.0, 5.0)
    assert s.std == (1.0, 1.0)


def test_dense_scaler_uses_population_std_and_transforms():
    raw = np.array([[1.0, 5.0], [3.0, 5.0]])
    s = DenseScaler.fit(raw)
    assert DenseScaler.fit(np.array([[0.0], [2.0]])).std == (1.0,)
    z = s.transform(raw)
    assert z.tolist() == [[-1.0, 0.0], [1.0, 0.0]]


def test_dense_scaler_rejects_zero_rows():
    with pytest.raises(ValueError):
        DenseScaler.fit(np.zeros((0, 8)))


def test_featurize_shapes_and_rows_match_scalar_functions():
    texts = ["a b", ""]
    feats = featurize(texts)
    assert feats.hashed.shape == (2, HASH_DIM)
    assert feats.hashed.dtype == np.float64
    assert feats.dense_raw.shape == (2, 8)
    idx, val = hashed_features("a b")
    row0 = feats.hashed.getrow(0)
    assert row0.indices.tolist() == idx
    assert row0.data.tolist() == val
    assert feats.hashed.getrow(1).nnz == 0
    for i, text in enumerate(texts):
        assert feats.dense_raw[i].tolist() == dense_raw(text)


def test_featurize_empty_input():
    feats = featurize([])
    assert feats.hashed.shape == (0, HASH_DIM)
    assert feats.dense_raw.shape == (0, 8)


def _doc_freq_matrix() -> sparse.csr_matrix:
    # document frequencies: index 9 -> 3 docs, 5 -> 3 docs, 2 -> 1 doc
    rows = [0, 0, 1, 1, 2, 2, 2]
    cols = [9, 5, 9, 5, 9, 5, 2]
    return sparse.csr_matrix((np.ones(7), (rows, cols)), shape=(3, HASH_DIM), dtype=np.float64)


def test_select_vocab_min_df_and_cap_ties_by_index():
    m = _doc_freq_matrix()
    assert select_vocab(m, min_df=2, max_size=1).tolist() == [5]
    assert select_vocab(m, min_df=2, max_size=10).tolist() == [5, 9]
    assert select_vocab(m, min_df=1, max_size=10).tolist() == [2, 5, 9]
    assert select_vocab(m, min_df=2, max_size=10).dtype == np.int64


def test_select_vocab_prefers_higher_df_over_smaller_index():
    rows = [0, 1, 2, 0, 1]
    cols = [9, 9, 9, 5, 5]
    m = sparse.csr_matrix((np.ones(5), (rows, cols)), shape=(3, HASH_DIM), dtype=np.float64)
    assert select_vocab(m, min_df=2, max_size=1).tolist() == [9]


def test_select_vocab_defaults_apply_min_df_five():
    rows = list(range(5)) + list(range(4))
    cols = [7] * 5 + [8] * 4
    m = sparse.csr_matrix((np.ones(9), (rows, cols)), shape=(5, HASH_DIM), dtype=np.float64)
    assert select_vocab(m).tolist() == [7]


def test_design_matrix_columns_and_global_feature():
    rows = [0, 1, 1]
    cols = [5, 9, 5]
    hashed = sparse.csr_matrix(
        (np.array([0.5, 0.25, 0.75]), (rows, cols)), shape=(2, HASH_DIM), dtype=np.float64
    )
    dense_z = np.arange(16, dtype=np.float64).reshape(2, 8) + 1.0
    vocab = np.array([5, 9], dtype=np.int64)
    design = design_matrix(hashed, dense_z, vocab)
    assert design.shape == (2, 10)
    assert design[:, 1].toarray().ravel().tolist() == hashed[:, 9].toarray().ravel().tolist()
    assert design[:, 2].toarray().ravel().tolist() == dense_z[:, 0].tolist()
    assert design[:, 0].toarray().ravel().tolist() == [0.5, 0.75]
    assert global_feature(1, vocab) == 9
    assert global_feature(2, vocab) == 262144
    assert global_feature(9, vocab) == 262144 + 7


@pytest.mark.parametrize("column", [-1, 10, 11])
def test_global_feature_rejects_columns_outside_the_design_matrix(column):
    with pytest.raises(ValueError, match=str(column)):
        global_feature(column, np.array([5, 9], dtype=np.int64))


def test_global_feature_accepts_first_and_last_column():
    vocab = np.array([5, 9], dtype=np.int64)
    assert global_feature(0, vocab) == 5
    assert global_feature(9, vocab) == HASH_DIM + 7


def test_design_matrix_rejects_wrong_dense_width():
    hashed = sparse.csr_matrix((2, HASH_DIM), dtype=np.float64)
    vocab = np.array([5], dtype=np.int64)
    for width in (7, 9):
        with pytest.raises(ValueError, match="dense"):
            design_matrix(hashed, np.zeros((2, width)), vocab)


def test_design_matrix_rejects_mismatched_row_count():
    hashed = sparse.csr_matrix((2, HASH_DIM), dtype=np.float64)
    vocab = np.array([5], dtype=np.int64)
    with pytest.raises(ValueError, match="rows"):
        design_matrix(hashed, np.zeros((3, 8)), vocab)


def test_design_matrix_rejects_one_dimensional_dense_block():
    hashed = sparse.csr_matrix((2, HASH_DIM), dtype=np.float64)
    with pytest.raises(ValueError, match="dense"):
        design_matrix(hashed, np.zeros(16), np.array([5], dtype=np.int64))


def test_prompt_sha256():
    assert (
        prompt_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
