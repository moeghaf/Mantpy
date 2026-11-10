"""Label corruption, context features, anomaly scores and reconstruction scoring."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantpy._analysis import (
    LabelReconstructionResult,
    _aligned,
    _categories,
    _ids,
    neighbourhood_features,
    reconstruct_labels,
)
from mantpy._constants import CELL_ECM_GRAPH_KEY, ECM_PATCHES_KEY
from mantpy._graph_layers import distance_weights, graph_layers


def _signal_categories(labels, categories) -> pd.Index:
    if categories is not None:
        return _categories([], categories)
    cats = _categories(labels)
    return cats[[not (isinstance(c, int | float | np.number) and c < 0) for c in cats]]


def _signal_rows(observed, probabilities, n_classes, categories, signal, ids):
    """Category index, the code of each observed label and the mask of the signal rows."""
    if categories is None and isinstance(probabilities, pd.DataFrame):
        categories = probabilities.columns
    cats = _categories([], range(n_classes) if categories is None else categories)
    if ids is None:
        if isinstance(observed, pd.Series):
            ids = observed.index
        elif isinstance(probabilities, pd.DataFrame):
            ids = probabilities.index
    row_ids = _ids(ids, len(observed), "ids")
    codes = cats.get_indexer(np.asarray(_aligned(observed, row_ids, "observed")))
    is_signal = codes >= 0
    if signal is not None:
        is_signal &= np.asarray(_aligned(signal, row_ids, "signal"), dtype=bool)
    return cats, codes, is_signal


# --------------------------------------------------------------------- corruption
@dataclass(frozen=True)
class LabelCorruptionResult:
    """Output of :func:`corrupt_labels`.

    ``labels`` are the labels after corruption, ``pristine`` the input labels
    and ``is_corrupted`` marks the replaced rows. ``block_id`` is the row of
    ``blocks`` that replaced a unit (-1 otherwise). ``blocks`` has one row per
    placed block: ``centre_x``, ``centre_y``, ``side``, ``new_label``,
    ``n_replaced``.
    """

    labels: np.ndarray
    pristine: np.ndarray
    is_corrupted: np.ndarray
    block_id: np.ndarray
    blocks: pd.DataFrame
    coordinates: np.ndarray
    n_requested: int
    random_state: int

    def in_block(self) -> np.ndarray:
        """Rows lying inside any block square, whether or not their label changed."""
        inside = np.zeros(len(self.labels), dtype=bool)
        for block in self.blocks.itertuples(index=False):
            half = block.side / 2.0
            inside |= (
                (np.abs(self.coordinates[:, 0] - block.centre_x) <= half)
                & (np.abs(self.coordinates[:, 1] - block.centre_y) <= half)
            )
        return inside

    def to_overlay(
        self,
        *,
        truth_key: str = "is_artifact",
        replacement_key: str = "ecm_cluster_artifact",
        reference_key: str = "ecm_cluster_pristine",
    ) -> dict[str, Any]:
        """Overlay accepted by :func:`mantpy.pp.apply_ecm_label_overlay`."""
        return {
            truth_key: self.is_corrupted.copy(),
            replacement_key: self.labels.copy(),
            reference_key: self.pristine.copy(),
            "artifact_boxes": [
                {
                    "center": [float(b.centre_x), float(b.centre_y)],
                    "half": float(b.side) / 2.0,
                    "new_cluster": b.new_label,
                    "n_replaced": int(b.n_replaced),
                }
                for b in self.blocks.itertuples(index=False)
            ],
        }


def corrupt_labels(
    labels,
    coordinates,
    *,
    fraction: float = 0.10,
    block_fraction: float = 0.5,
    block_sides: Sequence[float] = (50.0, 100.0, 150.0),
    min_block_units: int = 20,
    categories=None,
    signal=None,
    random_state: int = 0,
    on_shortfall: Literal["raise", "warn"] = "raise",
) -> LabelCorruptionResult:
    """Replace a seeded fraction of labels, in square blocks and as isolated units.

    A share ``block_fraction`` of the replacements is made inside square
    blocks: each block is centred near a random eligible unit and all its
    units receive one label drawn uniformly from ``categories`` (units that
    already carry it are left unchanged). The remaining replacements are
    isolated units, each given a different label drawn uniformly from the
    other categories. Exactly ``round(fraction * n_eligible)`` labels change.

    Parameters
    ----------
    labels
        One label per unit. Labels outside ``categories`` (for example -1
        background) are never replaced.
    coordinates
        ``(n, 2)`` unit positions, in the units of ``block_sides``.
    fraction
        Share of eligible labels to replace.
    block_fraction
        Share of the replacements made in blocks: 1 gives blocks only, 0
        isolated replacements only.
    block_sides
        Candidate block side lengths; one is drawn per block.
    min_block_units
        A block is placed only where it covers at least this many eligible
        units.
    categories
        Replacement label set, in order. By default the sorted labels present,
        without negative numbers. Pass it explicitly for a cohort, so that a
        sample lacking one class still draws from every class.
    signal
        Optional boolean mask further restricting the eligible units.
    random_state
        Seed. The same inputs and seed give the same corruption.
    on_shortfall
        ``"raise"`` or ``"warn"`` when the requested number cannot be placed.

    Returns
    -------
    :class:`LabelCorruptionResult`
    """
    from scipy.spatial import cKDTree

    if not 0 <= fraction <= 1 or not 0 <= block_fraction <= 1:
        raise ValueError("fraction and block_fraction must lie in [0, 1].")
    if on_shortfall not in {"raise", "warn"}:
        raise ValueError("on_shortfall must be 'raise' or 'warn'.")
    original = np.asarray(labels)
    if original.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    xy = np.asarray(coordinates, dtype=float)
    if xy.shape != (len(original), 2) or not np.isfinite(xy).all():
        raise ValueError("coordinates must be a finite (n, 2) array aligned with labels.")
    sides = np.asarray(block_sides, dtype=float)
    cats = _signal_categories(original, categories)
    n_labels = len(cats)
    if n_labels < 2:
        raise ValueError("At least two categories are needed to replace a label.")
    pristine = cats.get_indexer(original)
    eligible = pristine >= 0
    if signal is not None:
        mask = np.asarray(signal, dtype=bool)
        if mask.shape != eligible.shape:
            raise ValueError("signal must align with labels.")
        eligible &= mask

    # The order of random draws below is fixed: changing it changes every result.
    rng = np.random.default_rng(int(random_state))
    budget = int(round(fraction * int(eligible.sum())))
    block_budget = int(budget * block_fraction)
    isolated_budget = budget - block_budget
    corrupted = pristine.copy()
    replaced = np.zeros(len(pristine), dtype=bool)
    block_id = np.full(len(pristine), -1, dtype=int)
    blocks: list[dict[str, Any]] = []

    tries = 0
    while replaced.sum() < block_budget and tries < 200:
        tries += 1
        side = float(rng.choice(sides))
        half = side / 2.0
        candidates = np.flatnonzero(eligible & ~replaced)
        if candidates.size == 0:
            continue
        tree = cKDTree(xy[candidates])
        flips, new, centre = np.zeros(0, dtype=int), -1, (np.nan, np.nan)
        for _ in range(60):
            anchor = xy[rng.choice(candidates)]
            cx, cy = anchor[0] + rng.uniform(-half, half), anchor[1] + rng.uniform(-half, half)
            near = tree.query_ball_point([cx, cy], r=half * np.sqrt(2.0))
            near = candidates[np.asarray(near, dtype=int)] if len(near) else np.zeros(0, dtype=int)
            inside = near[(np.abs(xy[near, 0] - cx) <= half) & (np.abs(xy[near, 1] - cy) <= half)]
            if inside.size < min_block_units:
                continue
            new = int(rng.integers(0, n_labels))
            flips = inside[pristine[inside] != new]
            if flips.size == 0:
                continue
            centre = (float(cx), float(cy))
            break
        if flips.size == 0:
            continue
        remaining = block_budget - int(replaced.sum())
        if flips.size > remaining:
            flips = rng.choice(flips, size=remaining, replace=False)
        corrupted[flips] = new
        replaced[flips] = True
        block_id[flips] = len(blocks)
        blocks.append({
            "centre_x": centre[0], "centre_y": centre[1], "side": side,
            "new_label": cats[new], "n_replaced": int(flips.size),
        })

    pool = np.flatnonzero(eligible & ~replaced)
    if isolated_budget > 0 and pool.size:
        for i in rng.choice(pool, size=min(isolated_budget, pool.size), replace=False):
            others = np.arange(n_labels)
            others = others[others != pristine[i]]
            corrupted[i] = int(rng.choice(others))
            replaced[i] = True

    if int(replaced.sum()) != budget:
        message = f"Replaced {int(replaced.sum())} labels; {budget} were requested."
        if on_shortfall == "raise":
            raise ValueError(message + " Lower min_block_units or block_fraction.")
        warnings.warn(message, stacklevel=2)

    observed = original.astype(object) if original.dtype.kind in {"U", "S"} else original.copy()
    observed[replaced] = cats.to_numpy()[corrupted[replaced]]
    return LabelCorruptionResult(
        labels=observed,
        pristine=original.copy(),
        is_corrupted=replaced,
        block_id=block_id,
        blocks=pd.DataFrame(
            blocks, columns=["centre_x", "centre_y", "side", "new_label", "n_replaced"],
        ),
        coordinates=xy,
        n_requested=budget,
        random_state=int(random_state),
    )


# ----------------------------------------------------------------- anomaly scores
def label_anomaly_scores(
    observed,
    probabilities,
    *,
    categories=None,
    signal=None,
    score: str = "raw_neglogp",
    prior=None,
    temperature: float = 1.0,
    eps: float = 1e-12,
    ids=None,
) -> np.ndarray:
    """Score how unexpected each observed label is, without relabelling.

    The scores equal ``reconstruct_labels(...).anomaly_scores`` for the same
    ``score`` (``raw_neglogp``, ``minmax_neglogp``, ``one_minus_p`` or
    ``margin``). With ``prior`` (class frequencies of the training labels)
    the score is prior-adjusted,
    ``-log[P(observed | context) / prior(observed)]``, which removes the part
    of the score explained by label frequency alone.

    Parameters
    ----------
    observed, probabilities, categories, signal, temperature, eps, ids
        As in :func:`mantpy.tl.reconstruct_labels`.
    score
        Anomaly statistic.
    prior
        Class frequencies in ``categories`` order, or a Series / mapping
        keyed by category. Must sum to one. Only with ``score="raw_neglogp"``.

    Returns
    -------
    Float array with one score per row; rows that are not signal score 0.
    """
    result = reconstruct_labels(
        observed, probabilities, threshold=0.0, categories=categories, signal=signal,
        score=score, temperature=temperature, eps=eps, ids=ids,
    )
    scores = np.asarray(result.anomaly_scores, dtype=float)
    if prior is None:
        return scores
    if score != "raw_neglogp":
        raise ValueError("prior is only defined for score='raw_neglogp'.")
    n_classes = np.asarray(result.probabilities).shape[1]
    cats, codes, is_signal = _signal_rows(observed, probabilities, n_classes, categories, signal, ids)
    if isinstance(prior, pd.Series | Mapping):
        prior = pd.Series(prior).reindex(cats)
    frequencies = np.asarray(prior, dtype=float)
    if (
        frequencies.shape != (len(cats),) or not np.isfinite(frequencies).all()
        or (frequencies < 0).any() or not np.isclose(frequencies.sum(), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError("prior must give one non-negative frequency per category, summing to one.")
    rows = np.flatnonzero(is_signal)
    adjusted = scores.copy()
    adjusted[rows] = adjusted[rows] + np.log(np.maximum(frequencies[codes[rows]], eps))
    return adjusted


def fit_temperature(
    labels,
    probabilities,
    *,
    categories=None,
    signal=None,
    grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    eps: float = 1e-12,
    ids=None,
) -> float:
    """Choose the temperature that best calibrates class probabilities.

    Each temperature of ``grid`` is applied to ``probabilities`` as in
    :func:`reconstruct_labels`, and the one with the lowest mean negative
    log-probability of ``labels`` is returned (the first one on a tie). Use
    labels the classifier should predict, normally the training labels, and
    pass the result as ``temperature`` to :func:`label_anomaly_scores` or
    :func:`reconstruct_labels`. Labels absent from ``categories``
    (background) and rows outside ``signal`` are not used.
    """
    temperatures = np.asarray(grid, dtype=float)
    if (
        temperatures.ndim != 1 or not len(temperatures)
        or not np.isfinite(temperatures).all() or (temperatures <= 0).any()
    ):
        raise ValueError("grid must hold one or more finite, positive temperatures.")
    scores = [
        label_anomaly_scores(
            labels, probabilities, categories=categories, signal=signal,
            score="raw_neglogp", temperature=float(temperature), eps=eps, ids=ids,
        )
        for temperature in temperatures
    ]
    n_classes = np.asarray(probabilities).shape[1]
    is_signal = _signal_rows(labels, probabilities, n_classes, categories, signal, ids)[2]
    if not is_signal.any():
        raise ValueError("No label belongs to the categories, so no temperature can be fitted.")
    losses = [values[is_signal].mean() for values in scores]
    return float(temperatures[int(np.argmin(losses))])


# ------------------------------------------------------------- repeated passes
@dataclass(frozen=True)
class IterativeLabelReconstructionResult:
    """Output of :func:`reconstruct_labels_iteratively`.

    ``labels`` are the labels after the last pass and ``relabelled`` marks
    the rows whose label differs from the observed one. ``passes`` holds the
    :class:`LabelReconstructionResult` of every pass that was run.
    """

    labels: np.ndarray
    relabelled: np.ndarray
    passes: tuple[LabelReconstructionResult, ...]

    @property
    def n_passes(self) -> int:
        return len(self.passes)


def reconstruct_labels_iteratively(
    observed,
    predict,
    *,
    threshold: float,
    max_passes: int = 1,
    categories=None,
    signal=None,
    score: str = "minmax_neglogp",
    min_confidence: float = 0.0,
    coordinates=None,
    spatial_radius: float | None = None,
    min_neighbours: int = 0,
    majority_smoothing: bool = False,
    temperature: float = 1.0,
    eps: float = 1e-12,
    ids=None,
) -> IterativeLabelReconstructionResult:
    """Reconstruct labels in repeated passes, predicting again after each pass.

    ``predict(labels)`` returns the class probabilities of every row for the
    current labels. Each pass is one call of :func:`reconstruct_labels` with
    the options given here. The next pass predicts from the labels that call
    returned, so a classifier that reads the labels of neighbouring patches
    sees the corrections already made. The passes stop after ``max_passes``,
    or earlier when a pass changes no label. A classifier that does not read
    the labels needs one pass.

    When ``observed`` is a Series, or ``ids`` is given, ``predict`` receives a
    Series with those row IDs; otherwise it receives an array.
    """
    if not callable(predict):
        raise TypeError("predict must be a function from the current labels to class probabilities.")
    if isinstance(max_passes, bool) or not isinstance(max_passes, int | np.integer) or max_passes < 1:
        raise ValueError("max_passes must be a positive integer.")
    row_ids = None
    if ids is not None:
        row_ids = _ids(ids, len(observed), "ids")
    elif isinstance(observed, pd.Series):
        row_ids = observed.index
    start = np.asarray(observed if row_ids is None else _aligned(observed, row_ids, "observed"))
    current, passes = start, []
    for _ in range(int(max_passes)):
        given = current if row_ids is None else pd.Series(current, index=row_ids)
        step = reconstruct_labels(
            given, predict(given), threshold=threshold, categories=categories, signal=signal,
            score=score, min_confidence=min_confidence, coordinates=coordinates,
            spatial_radius=spatial_radius, min_neighbours=min_neighbours,
            majority_smoothing=majority_smoothing, temperature=temperature, eps=eps, ids=row_ids,
        )
        passes.append(step)
        current = np.asarray(step.labels)
        if not np.asarray(step.relabelled).any():
            break
    return IterativeLabelReconstructionResult(current, current != start, tuple(passes))


# ------------------------------------------------------------------------ scoring
@dataclass(frozen=True)
class LabelReconstructionScore:
    """Output of :func:`score_label_reconstruction`.

    ``metrics`` holds ``n_signal``, ``n_artifact``, ``n_relabelled``,
    ``corrupted_fraction``, ``before_accuracy``, ``after_accuracy``,
    ``artifact_recovered``, ``correction_precision``,
    ``pristine_relabel_rate`` and, when scores are supplied, ``roc_auc`` and
    ``pr_auc`` (suffixed with the score name for a mapping). ``per_class``
    has one row per category: ``n_signal``, ``n_corrupted``, ``n_restored``
    and ``recovery``.
    """

    metrics: dict[str, float]
    per_class: pd.DataFrame


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def score_label_reconstruction(
    pristine,
    observed,
    reconstructed,
    *,
    corrupted=None,
    scores=None,
    signal=None,
    categories=None,
) -> LabelReconstructionScore:
    """Score detection and correction of corrupted labels against the truth.

    Parameters
    ----------
    pristine, observed, reconstructed
        Row-aligned true labels, labels after corruption and labels after
        reconstruction.
    corrupted
        Boolean mask of the replaced labels. By default
        ``observed != pristine``.
    scores
        Anomaly scores (higher is more anomalous) for detecting the replaced
        labels: one array, or a mapping from a name to an array. They give
        ``roc_auc`` and ``pr_auc``.
    signal
        Optional boolean mask of the rows to score, combined with "the
        pristine label is one of ``categories``".
    categories
        Signal classes. By default the sorted pristine labels without
        negative numbers.

    Returns
    -------
    :class:`LabelReconstructionScore`. ``artifact_recovered`` is the share of
    replaced labels restored to the truth, ``pristine_relabel_rate`` the
    share of unreplaced labels made wrong, and ``correction_precision`` the
    share of changed labels that are now right.
    """
    truth, seen, final = (np.asarray(v) for v in (pristine, observed, reconstructed))
    if not (truth.ndim == 1 and truth.shape == seen.shape == final.shape):
        raise ValueError("pristine, observed and reconstructed must be aligned one-dimensional arrays.")
    cats = _signal_categories(truth, categories)
    mask = cats.get_indexer(truth) >= 0
    if signal is not None:
        extra = np.asarray(signal, dtype=bool)
        if extra.shape != mask.shape:
            raise ValueError("signal must align with the labels.")
        mask &= extra
    bad = seen != truth if corrupted is None else np.asarray(corrupted, dtype=bool)
    if bad.shape != mask.shape:
        raise ValueError("corrupted must align with the labels.")
    bad = bad & mask
    changed = (final != seen) & mask
    right = final == truth
    n_signal = int(mask.sum())
    metrics: dict[str, float] = {
        "n_signal": n_signal,
        "n_artifact": int(bad.sum()),
        "n_relabelled": int(changed.sum()),
        "corrupted_fraction": float(bad.sum() / n_signal) if n_signal else float("nan"),
        "before_accuracy": _mean((seen == truth)[mask]),
        "after_accuracy": _mean(right[mask]),
        "artifact_recovered": _mean(right[bad]),
        "correction_precision": _mean(right[changed]),
        "pristine_relabel_rate": _mean(~right[mask & ~bad]),
    }
    if scores is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        named = scores if isinstance(scores, Mapping) else {"": scores}
        y = bad[mask].astype(int)
        for name, values in named.items():
            values = np.asarray(values, dtype=float)
            if values.shape != mask.shape:
                raise ValueError("scores must align with the labels.")
            suffix = f"_{name}" if name else ""
            if y.min(initial=1) == y.max(initial=0):
                metrics[f"roc_auc{suffix}"] = metrics[f"pr_auc{suffix}"] = float("nan")
            else:
                metrics[f"roc_auc{suffix}"] = float(roc_auc_score(y, values[mask]))
                metrics[f"pr_auc{suffix}"] = float(average_precision_score(y, values[mask]))
    rows = []
    for category in cats:
        of_class = mask & (truth == category)
        hit = bad & of_class
        restored = int((right & hit).sum())
        rows.append({
            "n_signal": int(of_class.sum()), "n_corrupted": int(hit.sum()), "n_restored": restored,
            "recovery": restored / int(hit.sum()) if hit.any() else float("nan"),
        })
    return LabelReconstructionScore(metrics, pd.DataFrame(rows, index=cats))


# ----------------------------------------------------------------------- features
def context_features(
    adata: AnnData,
    *,
    context: Literal["cell", "ecm", "joint"] = "cell",
    hops: int = 1,
    normalization: str = "proportions",
    edge_weight: Literal["uniform", "inverse", "inverse_square", "gaussian"] = "uniform",
    sigma: float | Literal["median"] = "median",
    min_distance: float | None = None,
    cell_categories: Sequence | None = None,
    ecm_categories: Sequence | None = None,
    ecm_labels=None,
    graph_key: str = CELL_ECM_GRAPH_KEY,
    cell_attr: str = "cell_type",
    ecm_attr: str = "ecm_cluster",
    patches_key: str = ECM_PATCHES_KEY,
) -> pd.DataFrame:
    """Describe each ECM patch by the labels of its graph neighbours.

    ``"cell"`` context is the cell-type composition of the cells linked to
    the patch; ``"ecm"`` context is the cluster composition of its
    neighbouring patches; ``"joint"`` is both. The patch's own label is never
    used. With ``hops=2`` further blocks are appended: cells reached through
    cell-cell edges, patches reached through ECM-ECM edges, and patches
    sharing a cell with the patch.

    Parameters
    ----------
    adata
        Cells with ``uns['ecm_patches']`` and a built cell-ECM graph.
    context
        ``"cell"``, ``"ecm"`` or ``"joint"``.
    hops
        1 or 2.
    normalization
        ``"proportions"``, ``"counts"`` or ``"log1p_counts"``, per block.
    edge_weight, sigma, min_distance
        Distance kernel applied to each graph layer; see
        :func:`mantpy.gr.distance_weights`.
    cell_categories, ecm_categories
        Fixed column vocabularies. Pass them when samples are processed one
        at a time, so that all samples share the same columns. By default the
        sorted cell types of this sample and its non-negative cluster labels.
    ecm_labels
        Patch labels to use instead of the ``ecm_attr`` column, for example
        corrupted or partly reconstructed labels.

    Returns
    -------
    DataFrame with one row per patch, in patch-table order. Columns are
    prefixed ``cell:``, ``cell.cell:``, ``ecm:``, ``ecm.ecm:`` and
    ``cell.ecm:`` by block.
    """
    if context not in {"cell", "ecm", "joint"}:
        raise ValueError("context must be 'cell', 'ecm' or 'joint'.")
    if hops not in {1, 2}:
        raise ValueError("hops must be 1 or 2.")
    patches = adata.uns[patches_key]
    layers = graph_layers(adata, graph_key=graph_key)

    def weighted(connectivity, distances, name):
        if edge_weight == "uniform":
            return connectivity
        if distances is None:
            raise ValueError(f"The {name} layer stores no edge lengths; use edge_weight='uniform'.")
        return distance_weights(distances, kernel=edge_weight, sigma=sigma, min_distance=min_distance)

    cell_ecm = weighted(layers.cell_ecm, layers.cell_ecm_distances, "cell-ECM")
    cells = np.asarray([f"cell:{i}" for i in range(adata.n_obs)])
    nodes = np.asarray([f"ecm:{i}" for i in range(len(patches))])
    blocks = []

    def block(adjacency, labels, categories, sources, prefix, via=None):
        blocks.append(neighbourhood_features(
            adjacency, labels, categories=categories, target_ids=nodes, source_ids=sources,
            normalization=normalization, hops=2 if via is not None else 1, via=via,
            exclude_self=True, prefix=prefix,
        ))

    if context in {"cell", "joint"}:
        types = adata.obs[cell_attr].astype(str).to_numpy()
        vocabulary = sorted(set(types)) if cell_categories is None else [str(c) for c in cell_categories]
        block(cell_ecm.T, types, vocabulary, cells, "cell:")
        if hops == 2:
            block(cell_ecm.T, types, vocabulary, cells, "cell.cell:",
                  weighted(layers.cell, layers.cell_distances, "cell-cell"))
    if context in {"ecm", "joint"}:
        labels = np.asarray(patches[ecm_attr] if ecm_labels is None else ecm_labels)
        if len(labels) != len(patches):
            raise ValueError("ecm_labels must have one label per patch.")
        vocabulary = list(_signal_categories(labels, ecm_categories))
        ecm = weighted(layers.ecm, layers.ecm_distances, "ECM-ECM")
        block(ecm, labels, vocabulary, nodes, "ecm:")
        if hops == 2:
            block(ecm, labels, vocabulary, nodes, "ecm.ecm:", ecm)
            block(cell_ecm.T, labels, vocabulary, nodes, "cell.ecm:", cell_ecm)
    result = pd.concat(blocks, axis=1)
    result.index = pd.Index(patches.index)
    return result
