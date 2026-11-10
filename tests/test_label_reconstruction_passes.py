"""Tests for ``mt.tl.fit_temperature`` and ``mt.tl.reconstruct_labels_iteratively``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mantpy as mt

GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _calibration_case(sharpness: float, seed: int = 0, n: int = 3000):
    """Labels drawn from known class probabilities, and those probabilities raised to ``sharpness``."""
    rng = np.random.default_rng(seed)
    true = rng.dirichlet(np.ones(3), size=n)
    labels = (rng.random(n)[:, None] > np.cumsum(true, axis=1)).sum(axis=1).clip(max=2)
    given = true**sharpness
    return labels, given / given.sum(axis=1, keepdims=True)


def _reference_temperature(probabilities, labels, grid=GRID):
    """Temperature with the lowest mean negative log-probability, written out directly."""

    def loss(temperature):
        scaled = probabilities ** (1 / temperature)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.maximum(scaled[np.arange(len(labels)), labels], 1e-12)))

    return min(grid, key=loss)


# ------------------------------------------------------------------ fit_temperature
@pytest.mark.parametrize(("sharpness", "expected"), [(2.0, 2.0), (1.0, 1.0), (0.5, 0.5)])
def test_fit_temperature_recovers_the_known_temperature(sharpness, expected):
    labels, probabilities = _calibration_case(sharpness)
    fitted = mt.tl.fit_temperature(labels, probabilities)
    assert fitted == expected
    assert fitted == _reference_temperature(probabilities, labels)


@pytest.mark.parametrize("seed", range(5))
def test_fit_temperature_matches_the_direct_calculation(seed):
    labels, probabilities = _calibration_case(1.4, seed=seed, n=400)
    assert mt.tl.fit_temperature(labels, probabilities) == _reference_temperature(probabilities, labels)


def test_fit_temperature_returns_a_float_from_the_grid():
    labels, probabilities = _calibration_case(2.0)
    fitted = mt.tl.fit_temperature(labels, probabilities, grid=[1.0, 4.0])
    assert isinstance(fitted, float)
    assert fitted in (1.0, 4.0)


def test_fit_temperature_returns_the_first_temperature_on_a_tie():
    labels = np.array([0, 1, 2, 0])
    uniform = np.full((4, 3), 1 / 3)
    assert mt.tl.fit_temperature(labels, uniform, grid=(2.0, 1.0, 0.5)) == 2.0


def test_fit_temperature_ignores_background_labels_and_rows_outside_signal():
    labels, probabilities = _calibration_case(2.0, n=600)
    kept = np.ones(len(labels), dtype=bool)
    kept[::3] = False
    with_background = labels.copy()
    with_background[~kept] = -1
    # The dropped rows favour a low temperature; they must not move the result.
    altered = probabilities.copy()
    altered[~kept] = np.eye(3)[labels[~kept]] * 0.4 + 0.2
    expected = _reference_temperature(probabilities[kept], labels[kept])
    assert mt.tl.fit_temperature(with_background, altered, categories=range(3)) == expected
    assert mt.tl.fit_temperature(labels, altered, signal=kept) == expected


def test_fit_temperature_reads_named_categories_from_a_table():
    labels, probabilities = _calibration_case(2.0, n=500)
    index = [f"p{i}" for i in range(len(labels))]
    names = np.array(["alveolar", "wall", "adventitial"])
    table = pd.DataFrame(probabilities, index=index, columns=names)
    series = pd.Series(names[labels], index=index)
    assert mt.tl.fit_temperature(series, table) == mt.tl.fit_temperature(labels, probabilities)
    # Rows are matched by their IDs, not by position.
    assert mt.tl.fit_temperature(series.iloc[::-1], table) == mt.tl.fit_temperature(labels, probabilities)


@pytest.mark.parametrize("grid", [(), (0.0, 1.0), (-1.0,), (np.nan, 1.0), (np.inf,), [[1.0, 2.0]]])
def test_fit_temperature_rejects_an_invalid_grid(grid):
    labels, probabilities = _calibration_case(1.0, n=30)
    with pytest.raises(ValueError, match="grid"):
        mt.tl.fit_temperature(labels, probabilities, grid=grid)


def test_fit_temperature_needs_a_label_from_the_categories():
    _, probabilities = _calibration_case(1.0, n=30)
    with pytest.raises(ValueError):
        mt.tl.fit_temperature(np.full(30, -1), probabilities, categories=range(3))


# --------------------------------------------------- reconstruct_labels_iteratively
def _chain(seed: int = 0, n: int = 90, wrong: float = 0.2):
    """Three blocks of labels along a line, with a share of them replaced at random."""
    rng = np.random.default_rng(seed)
    truth = np.repeat([0, 1, 2], n // 3)
    observed = truth.copy()
    rows = rng.choice(n, size=int(wrong * n), replace=False)
    observed[rows] = (truth[rows] + rng.integers(1, 3, size=len(rows))) % 3
    return truth, observed


def _neighbour_model(reach: int = 3):
    """Class probabilities from the labels of the rows up to ``reach`` steps away, not the row itself."""

    def predict(labels):
        onehot = np.eye(3)[np.asarray(labels, dtype=int)]
        counts = np.full_like(onehot, 0.1)
        for step in range(1, reach + 1):
            counts[step:] += onehot[:-step]
            counts[:-step] += onehot[step:]
        return counts / counts.sum(axis=1, keepdims=True)

    return predict


def _manual_passes(observed, predict, max_passes, **options):
    current, n_passes = np.asarray(observed).copy(), 0
    for _ in range(max_passes):
        step = mt.tl.reconstruct_labels(current, predict(current), **options)
        n_passes += 1
        current = np.asarray(step.labels)
        if not np.asarray(step.relabelled).any():
            break
    return current, n_passes


def test_one_pass_equals_reconstruct_labels():
    _, observed = _chain()
    predict = _neighbour_model()
    options = {"threshold": 0.6, "score": "one_minus_p", "categories": range(3)}
    single = mt.tl.reconstruct_labels(observed, predict(observed), **options)
    result = mt.tl.reconstruct_labels_iteratively(observed, predict, **options)
    assert result.n_passes == 1
    np.testing.assert_array_equal(result.labels, single.labels)
    np.testing.assert_array_equal(result.relabelled, single.relabelled)
    np.testing.assert_array_equal(result.passes[0].anomaly_scores, single.anomaly_scores)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "options",
    [
        {"threshold": 0.6, "score": "one_minus_p"},
        {"threshold": 0.3, "score": "minmax_neglogp", "min_confidence": 0.5},
        {"threshold": 1.0, "score": "raw_neglogp", "temperature": 1.5},
    ],
)
def test_repeated_passes_equal_the_loop_written_out(seed, options):
    _, observed = _chain(seed)
    predict = _neighbour_model()
    options = dict(options, categories=range(3))
    expected, n_passes = _manual_passes(observed, predict, 4, **options)
    result = mt.tl.reconstruct_labels_iteratively(observed, predict, max_passes=4, **options)
    np.testing.assert_array_equal(result.labels, expected)
    np.testing.assert_array_equal(result.relabelled, expected != observed)
    assert result.n_passes == n_passes == len(result.passes)


def test_later_passes_use_the_corrected_labels():
    truth, observed = _chain(seed=1)
    predict = _neighbour_model()
    options = {"threshold": 0.6, "score": "one_minus_p", "categories": range(3)}
    one = mt.tl.reconstruct_labels_iteratively(observed, predict, max_passes=1, **options)
    many = mt.tl.reconstruct_labels_iteratively(observed, predict, max_passes=6, **options)
    assert many.n_passes > 1
    assert np.asarray(many.passes[1].relabelled).any()
    assert (many.labels == truth).mean() > (one.labels == truth).mean() > (observed == truth).mean()


def test_passes_stop_when_no_label_changes():
    _, observed = _chain()
    fixed = _neighbour_model()(observed)  # a classifier that does not read the labels
    options = {"threshold": 0.6, "score": "one_minus_p", "categories": range(3)}
    result = mt.tl.reconstruct_labels_iteratively(observed, lambda labels: fixed, max_passes=10, **options)
    assert result.relabelled.any()
    assert result.n_passes == 2
    assert not np.asarray(result.passes[-1].relabelled).any()

    unchanged = mt.tl.reconstruct_labels_iteratively(
        observed, lambda labels: fixed, max_passes=10, threshold=1.0, score="one_minus_p", categories=range(3)
    )
    assert unchanged.n_passes == 1
    assert not unchanged.relabelled.any()
    np.testing.assert_array_equal(unchanged.labels, observed)


def test_the_observed_labels_are_not_modified():
    _, observed = _chain()
    before = observed.copy()
    mt.tl.reconstruct_labels_iteratively(
        observed, _neighbour_model(), max_passes=3, threshold=0.6, score="one_minus_p", categories=range(3)
    )
    np.testing.assert_array_equal(observed, before)


def test_background_labels_are_kept():
    _, observed = _chain()
    observed = observed.copy()
    observed[::7] = -1
    model = _neighbour_model()

    def predict(labels):
        values = np.asarray(labels).copy()
        values[values < 0] = 0
        return model(values)

    result = mt.tl.reconstruct_labels_iteratively(
        observed, predict, max_passes=3, threshold=0.6, score="one_minus_p", categories=range(3)
    )
    assert (result.labels[::7] == -1).all()
    assert not result.relabelled[::7].any()


def test_predict_receives_a_series_when_rows_have_ids():
    _, observed = _chain()
    model = _neighbour_model()
    seen = []

    def predict(labels):
        seen.append(labels)
        return model(labels)

    options = {"max_passes": 2, "threshold": 0.6, "score": "one_minus_p", "categories": range(3)}
    index = [f"patch_{i}" for i in range(len(observed))]
    from_series = mt.tl.reconstruct_labels_iteratively(pd.Series(observed, index=index), predict, **options)
    assert all(isinstance(labels, pd.Series) and list(labels.index) == index for labels in seen)

    seen.clear()
    from_ids = mt.tl.reconstruct_labels_iteratively(observed, predict, ids=index, **options)
    assert all(isinstance(labels, pd.Series) and list(labels.index) == index for labels in seen)

    seen.clear()
    from_array = mt.tl.reconstruct_labels_iteratively(observed, predict, **options)
    assert all(isinstance(labels, np.ndarray) for labels in seen)

    np.testing.assert_array_equal(from_series.labels, from_array.labels)
    np.testing.assert_array_equal(from_ids.labels, from_array.labels)


def test_predict_may_return_a_table():
    _, observed = _chain()
    model = _neighbour_model()
    index = [f"patch_{i}" for i in range(len(observed))]
    options = {"max_passes": 3, "threshold": 0.6, "score": "one_minus_p"}

    def predict(labels):
        return pd.DataFrame(model(labels), index=labels.index, columns=[0, 1, 2])

    from_table = mt.tl.reconstruct_labels_iteratively(pd.Series(observed, index=index), predict, **options)
    from_array = mt.tl.reconstruct_labels_iteratively(observed, model, categories=range(3), **options)
    np.testing.assert_array_equal(from_table.labels, from_array.labels)


def test_invalid_arguments_raise():
    _, observed = _chain()
    with pytest.raises(TypeError, match="predict"):
        mt.tl.reconstruct_labels_iteratively(observed, np.zeros((len(observed), 3)), threshold=0.5)
    for max_passes in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_passes"):
            mt.tl.reconstruct_labels_iteratively(
                observed, _neighbour_model(), threshold=0.5, max_passes=max_passes
            )
