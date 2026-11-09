from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data
nn = pytest.importorskip("mantpy.nn")
node_classifier = pytest.importorskip("mantpy.nn._node_classifier")

PREDICTION_COLUMNS = [
    "estimate",
    "seed",
    "unit_id",
    "fold",
    "true_class_idx",
    "predicted_class_idx",
    "correct",
    "probability_class_0",
    "probability_class_1",
]
ATTRIBUTION_COLUMNS = [
    "graph_index",
    "sample_id",
    "group_id",
    "class_idx",
    "fold",
    "node_id",
    "x",
    "y",
    "cluster",
    "node_importance",
    "max_abs_delta",
]


def _graph(label: int, clusters, *, n_features: int = 4, roi_id: str | None = None, layout: bool = False):
    """One-hot cluster graph on a chain; ``layout`` adds node ids and coordinates."""
    cluster_array = np.asarray(clusters, dtype=np.int64)
    n_nodes = len(cluster_array)
    chain = np.arange(n_nodes - 1)
    edges = np.stack([np.concatenate([chain, chain + 1]), np.concatenate([chain + 1, chain])])
    graph = Data(
        x=torch.eye(n_features, dtype=torch.float32)[torch.from_numpy(cluster_array)],
        edge_index=torch.from_numpy(edges).long(),
    )
    graph.cluster = torch.from_numpy(cluster_array)
    graph.y = torch.tensor(label)
    if roi_id is not None:
        graph.roi_id = roi_id
    if layout:
        # Ids skip positions, as they do after background nodes are dropped.
        graph.node_ids = [f"ecm_{2 * node + 1}" for node in range(n_nodes)]
        graph.pos = torch.tensor([[5.0 + 10.0 * node, 15.0 + 20.0 * node] for node in range(n_nodes)])
    return graph


# ---------------------------------------------------------------------------
# A5: NodeClassifier.composition_baseline
# ---------------------------------------------------------------------------
GROUP_ORDER = ["m5", "m2", "m7", "m0", "m3", "m6", "m1", "m4"]  # first-appearance order differs from sorted order
C_BY_GROUP = {"m0": 1.0, "m1": 0.1, "m2": 10.0, "m3": 1.0, "m4": 10.0, "m5": 0.5, "m6": 2.0, "m7": 10.0}
CLUSTER_IDS = [0, 2, 5]  # deliberately not contiguous


def _two_class_cohort():
    """Eight groups with two ROIs each; ROIs of one group are not adjacent in graph order."""
    rng = np.random.default_rng(7)
    graphs, labels, groups, clusters = [], [], [], []
    for roi in range(2):
        for group in GROUP_ORDER:
            label = int(group[1]) % 2
            weights = [0.5, 0.3, 0.2] if label == 0 else [0.3, 0.3, 0.4]
            codes = rng.choice(3, size=int(rng.integers(8, 14)), p=weights)
            graphs.append(_graph(label, codes, n_features=3, roi_id=f"{group}_roi{roi}"))
            graphs[-1].cluster = torch.from_numpy(np.asarray(CLUSTER_IDS)[codes])
            clusters.append(np.asarray(CLUSTER_IDS)[codes])
            labels.append(label)
            groups.append(group)
    return graphs, np.asarray(labels), np.asarray(groups), clusters


def _leave_one_group_out(groups):
    return [(np.flatnonzero(groups != group), np.flatnonzero(groups == group)) for group in sorted(set(groups))]


def _fractions(clusters):
    return np.stack([np.asarray([(values == k).sum() for k in CLUSTER_IDS]) / len(values) for values in clusters])


def _fit_two_class(*, with_groups: bool = True, folds=None):
    graphs, labels, groups, clusters = _two_class_cohort()
    folds = _leave_one_group_out(groups) if folds is None else folds
    classifier = nn.NodeClassifier(graphs, labels=labels, n_classes=2, hidden_dim=4, n_layers=1, device="cpu")
    classifier.fit(splits=folds, groups=groups if with_groups else None, epochs=1)
    return classifier, labels, groups, clusters, folds


@pytest.fixture(scope="module")
def two_class():
    return _fit_two_class()


def _hand_loop(fractions, labels, folds, fold_c, *, standardize=True):
    expected = np.zeros((len(labels), 2))
    for (train, test), c in zip(folds, fold_c, strict=True):
        train_x, test_x = fractions[train], fractions[test]
        if standardize:
            scaler = StandardScaler().fit(train_x)
            train_x, test_x = scaler.transform(train_x), scaler.transform(test_x)
        model = LogisticRegression(C=c, max_iter=10000).fit(train_x, labels[train])
        expected[test] = model.predict_proba(test_x)
    return expected


def test_composition_baseline_equals_hand_loop(two_class):
    classifier, labels, groups, clusters, folds = two_class
    fractions = _fractions(clusters)
    expected = _hand_loop(fractions, labels, folds, [C_BY_GROUP[groups[test[0]]] for _, test in folds])

    graph_table = classifier.composition_baseline(C=C_BY_GROUP, max_iter=10000, level="graph")
    group_table = classifier.composition_baseline(C=C_BY_GROUP, max_iter=10000)

    assert graph_table["unit_id"].tolist() == [str(index) for index in range(len(labels))]
    np.testing.assert_allclose(
        graph_table[["probability_class_0", "probability_class_1"]].to_numpy(), expected, rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(graph_table["true_class_idx"], labels)
    np.testing.assert_array_equal(graph_table["predicted_class_idx"], expected.argmax(axis=1))
    np.testing.assert_array_equal(graph_table["correct"], expected.argmax(axis=1) == labels)

    # The notebook's group mean: pandas groupby over the graph probabilities.
    by_group = pd.DataFrame(expected).groupby(groups).mean()
    assert group_table["unit_id"].tolist() == GROUP_ORDER  # first appearance, not sorted
    np.testing.assert_allclose(
        group_table[["probability_class_0", "probability_class_1"]].to_numpy(),
        by_group.loc[group_table["unit_id"]].to_numpy(),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(group_table["true_class_idx"], [int(group[1]) % 2 for group in GROUP_ORDER])

    # The per-group C values matter here: a single C gives different probabilities.
    uniform = classifier.composition_baseline(C=1.0, level="graph")
    assert np.abs(uniform["probability_class_1"].to_numpy() - expected[:, 1]).max() > 1e-3


def test_composition_baseline_uses_the_fitted_folds_and_per_group_c(two_class, monkeypatch):
    classifier, labels, groups, clusters, folds = two_class
    fractions = _fractions(clusters)
    calls = []

    class RecordingRegression(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            calls.append({"C": self.C, "max_iter": self.max_iter, "X": np.array(X), "y": np.array(y)})
            return super().fit(X, y, sample_weight)

    monkeypatch.setattr(node_classifier, "LogisticRegression", RecordingRegression)
    table = classifier.composition_baseline(C=C_BY_GROUP, max_iter=321, level="graph")

    # One regression per fitted fold, in fit order, trained on exactly that fold's training graphs.
    assert len(calls) == len(classifier._splits) == len(folds) == 8
    for call, (train, test), (fitted_train, fitted_test) in zip(calls, folds, classifier._splits, strict=True):
        np.testing.assert_array_equal(fitted_train, train)
        np.testing.assert_array_equal(fitted_test, test)
        held_out = set(groups[test])
        assert len(held_out) == 1
        assert call["C"] == C_BY_GROUP[held_out.pop()]
        assert call["max_iter"] == 321
        np.testing.assert_array_equal(call["y"], labels[train])
        np.testing.assert_array_equal(call["X"], StandardScaler().fit(fractions[train]).transform(fractions[train]))

    assignments = classifier.split_assignments()
    np.testing.assert_array_equal(table["fold"], assignments["fold"])
    group_table = classifier.composition_baseline(C=C_BY_GROUP)
    expected_fold = assignments.groupby("group_id")["fold"].first()
    np.testing.assert_array_equal(group_table["fold"], expected_fold.loc[group_table["unit_id"]])


def test_composition_baseline_c_forms_agree(two_class):
    classifier, _, groups, _, folds = two_class
    for level in ("graph", "group"):
        scalar = classifier.composition_baseline(C=2.0, level=level)
        sequence = classifier.composition_baseline(C=[2.0] * len(folds), level=level)
        array = classifier.composition_baseline(C=np.full(len(folds), 2.0), level=level)
        mapping = classifier.composition_baseline(C=dict.fromkeys(groups, 2.0), level=level)
        pd.testing.assert_frame_equal(scalar, sequence)
        pd.testing.assert_frame_equal(scalar, array)
        pd.testing.assert_frame_equal(scalar, mapping)

    # A sequence is read in fold order, a mapping by held-out group.
    fold_c = [C_BY_GROUP[groups[test[0]]] for _, test in folds]
    pd.testing.assert_frame_equal(
        classifier.composition_baseline(C=fold_c),
        classifier.composition_baseline(C=C_BY_GROUP),
    )


def test_composition_baseline_default_features_are_cluster_fractions(two_class):
    classifier, labels, _, clusters, folds = two_class
    fractions = _fractions(clusters)
    assert fractions.shape == (16, 3)

    default = classifier.composition_baseline(level="graph")
    pd.testing.assert_frame_equal(default, classifier.composition_baseline(features=fractions, level="graph"))

    unscaled = classifier.composition_baseline(level="graph", standardize=False, C=3.0)
    expected = _hand_loop(fractions, labels, folds, [3.0] * len(folds), standardize=False)
    np.testing.assert_allclose(
        unscaled[["probability_class_0", "probability_class_1"]].to_numpy(), expected, rtol=0, atol=1e-12
    )
    assert not np.allclose(unscaled["probability_class_1"], default["probability_class_1"])


def test_composition_baseline_dataframe_aligned_by_roi_id(two_class):
    classifier, _, _, clusters, _ = two_class
    roi_ids = [graph.roi_id for graph in classifier._graphs]
    frame = pd.DataFrame(_fractions(clusters), index=roi_ids, columns=["ECM0", "ECM2", "ECM5"])
    shuffled = frame.sample(frac=1.0, random_state=3)
    assert shuffled.index.tolist() != roi_ids

    expected = classifier.composition_baseline(C=C_BY_GROUP)
    pd.testing.assert_frame_equal(classifier.composition_baseline(features=shuffled, C=C_BY_GROUP), expected)

    with pytest.raises(ValueError, match="features must contain every requested ID exactly once"):
        classifier.composition_baseline(features=shuffled.iloc[1:])


def test_composition_baseline_dataframe_falls_back_to_graph_index_strings():
    labels = np.arange(8) % 2
    graphs = [
        _graph(label, [label, label, 1 - label, index % 3 % 2], n_features=2) for index, label in enumerate(labels)
    ]
    folds = [(np.setdiff1d(np.arange(8), test), np.asarray(test)) for test in ([0, 1], [2, 3], [4, 5], [6, 7])]
    classifier = nn.NodeClassifier(graphs, labels=labels, n_classes=2, hidden_dim=4, n_layers=1, device="cpu")
    classifier.fit(splits=folds, epochs=1)
    values = np.random.default_rng(0).normal(size=(8, 2)) + labels[:, None]
    frame = pd.DataFrame(values, index=[str(index) for index in range(8)]).iloc[::-1]

    positional = classifier.composition_baseline(features=values, level="graph")
    pd.testing.assert_frame_equal(classifier.composition_baseline(features=frame, level="graph"), positional)
    assert positional["unit_id"].tolist() == [str(index) for index in range(8)]


def test_composition_baseline_schema_matches_prediction_table(two_class):
    classifier, _, _, _, _ = two_class
    before = classifier.prediction_table(level="group")
    for level in ("group", "graph"):
        reference = classifier.prediction_table(level=level).query("estimate == 'ensemble'").reset_index(drop=True)
        table = classifier.composition_baseline(C=C_BY_GROUP, level=level)

        assert list(table.columns) == list(reference.columns) == PREDICTION_COLUMNS
        assert isinstance(table.index, pd.RangeIndex)
        assert set(table["estimate"]) == {"composition_baseline"}
        assert table["seed"].isna().all()
        for column in ("unit_id", "fold", "true_class_idx"):
            assert table[column].tolist() == reference[column].tolist()
        for column in ("fold", "true_class_idx", "predicted_class_idx", "correct"):
            assert table[column].dtype == reference[column].dtype
        probabilities = table[["probability_class_0", "probability_class_1"]].to_numpy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(table["predicted_class_idx"], probabilities.argmax(axis=1))
        np.testing.assert_array_equal(table["correct"], table["predicted_class_idx"] == table["true_class_idx"])

    # Nothing is written to the classifier.
    pd.testing.assert_frame_equal(classifier.prediction_table(level="group"), before)


def test_composition_baseline_requires_fit():
    graphs, labels, _, _ = _two_class_cohort()
    classifier = nn.NodeClassifier(graphs, labels=labels, n_classes=2, device="cpu")
    with pytest.raises(RuntimeError, match=r"Call \.fit\(\) first\."):
        classifier.composition_baseline()


def test_composition_baseline_errors(two_class):
    classifier, _, groups, clusters, folds = two_class
    fractions = _fractions(clusters)

    with pytest.raises(ValueError, match="C has 3 values but the classifier has 8 folds"):
        classifier.composition_baseline(C=[1.0, 1.0, 1.0])
    with pytest.raises(KeyError, match="C has no value for held-out group 'm0'"):
        classifier.composition_baseline(C={group: 1.0 for group in groups if group != "m0"})
    for bad in (0.0, -1.0, np.nan, np.inf, [1.0] * 7 + [0.0], {**C_BY_GROUP, "m3": -2.0}):
        with pytest.raises(ValueError, match="C must be finite and positive"):
            classifier.composition_baseline(C=bad)
    with pytest.raises(ValueError, match="features has 15 rows; expected 16"):
        classifier.composition_baseline(features=fractions[:-1])
    broken = fractions.copy()
    broken[2, 1] = np.nan
    with pytest.raises(ValueError, match="features must be a finite graph-by-feature matrix"):
        classifier.composition_baseline(features=broken)
    with pytest.raises(ValueError, match="level must be 'graph' or 'group'"):
        classifier.composition_baseline(level="mouse")

    # Without groups at fit: no group level and no mapping.
    ungrouped = _fit_two_class(with_groups=False)[0]
    with pytest.raises(ValueError, match=r"Group predictions require groups passed to fit\(\)\."):
        ungrouped.composition_baseline()
    with pytest.raises(ValueError, match=r"C as a mapping requires groups passed to fit\(\)\."):
        ungrouped.composition_baseline(C=C_BY_GROUP, level="graph")
    assert len(ungrouped.composition_baseline(C=[1.0] * len(folds), level="graph")) == 16

    # A fold that holds out two groups cannot be keyed by one group.
    paired = [
        (np.flatnonzero(~np.isin(groups, pair)), np.flatnonzero(np.isin(groups, pair)))
        for pair in (["m0", "m1"], ["m2", "m3"], ["m4", "m5"], ["m6", "m7"])
    ]
    two_per_fold = _fit_two_class(folds=paired)[0]
    with pytest.raises(ValueError, match="C as a mapping needs one held-out group per fold; fold 1 holds 2"):
        two_per_fold.composition_baseline(C=C_BY_GROUP)
    assert len(two_per_fold.composition_baseline(C=[1.0, 2.0, 3.0, 4.0])) == 8


# ---------------------------------------------------------------------------
# A6: NodeClassifier.attribution_table
# ---------------------------------------------------------------------------
def _four_class_cohort():
    graphs, labels, groups = [], [], []
    for label in range(4):
        for mouse in range(2):
            group = f"class{label}_mouse{mouse}"
            for roi in range(2):
                other = (label + mouse + roi + 1) % 4
                clusters = [label, label, label, other] + [other] * roi  # four or five nodes
                graphs.append(_graph(label, clusters, roi_id=f"{group}_roi{roi}", layout=True))
                labels.append(label)
                groups.append(group)
    return graphs, np.asarray(labels), np.asarray(groups)


@pytest.fixture(scope="module")
def four_class():
    graphs, labels, groups = _four_class_cohort()
    classifier = nn.NodeClassifier(graphs, labels=labels, n_classes=4, hidden_dim=4, n_layers=1, device="cpu")
    classifier.fit(n_splits=2, groups=groups, seeds=[0, 1], epochs=2)
    return classifier, graphs, labels, groups


def test_attribution_table_equals_integrated_gradients(four_class):
    classifier, graphs, _, _ = four_class
    table = classifier.attribution_table(target_class=1, n_steps=5, internal_batch_size=1)

    assert list(table.columns) == ATTRIBUTION_COLUMNS
    assert isinstance(table.index, pd.RangeIndex)
    assert table["node_importance"].dtype == np.float32
    # One row per node, graph after graph in graph order.
    sizes = [graph.num_nodes for graph in graphs]
    np.testing.assert_array_equal(table["graph_index"], np.repeat(np.arange(len(graphs)), sizes))
    assert table["node_id"].tolist() == [node for graph in graphs for node in graph.node_ids]

    for index, graph in enumerate(graphs):
        rows = table[table["graph_index"] == index]
        details = classifier.integrated_gradients(index, 1, n_steps=5, internal_batch_size=1, return_details=True)
        importance, _ = classifier.integrated_gradients(index, 1, n_steps=5, internal_batch_size=1)
        assert rows["node_id"].tolist() == list(details.node_ids) == graph.node_ids
        np.testing.assert_array_equal(rows["node_importance"].to_numpy(), details.node_importance)
        np.testing.assert_array_equal(rows["node_importance"].to_numpy(), importance)
        assert len(details.convergence_delta) == 2  # one per seed
        np.testing.assert_array_equal(rows["max_abs_delta"], np.abs(details.convergence_delta).max())
    assert (table["node_importance"] > 0).any()


def test_attribution_table_single_graph_and_default_step_arguments(four_class):
    classifier, graphs, _, _ = four_class
    table = classifier.attribution_table(3, indices=[5])
    details = classifier.integrated_gradients(5, 3, return_details=True)  # n_steps=50, no internal batching

    assert len(table) == graphs[5].num_nodes
    assert set(table["graph_index"]) == {5}
    assert table["node_id"].tolist() == list(details.node_ids)
    np.testing.assert_array_equal(table["node_importance"].to_numpy(), details.node_importance)


def test_attribution_table_carries_graph_attributes(four_class):
    classifier, graphs, labels, groups = four_class
    table = classifier.attribution_table(target_class=0, n_steps=2)
    assignments = classifier.split_assignments().set_index("graph_index")

    for index, graph in enumerate(graphs):
        rows = table[table["graph_index"] == index]
        assert set(rows["sample_id"]) == {graph.roi_id}
        assert set(rows["group_id"]) == {groups[index]} == {assignments.loc[index, "group_id"]}
        assert set(rows["class_idx"]) == {labels[index]}
        assert set(rows["fold"]) == {assignments.loc[index, "fold"]}
        np.testing.assert_array_equal(rows[["x", "y"]].to_numpy(), graph.pos.numpy())
        np.testing.assert_array_equal(rows["cluster"].to_numpy(), graph.cluster.numpy())
    assert table["x"].dtype == table["y"].dtype == np.float64
    assert table["max_abs_delta"].dtype == np.float64
    assert np.issubdtype(table["cluster"].dtype, np.integer)
    assert np.issubdtype(table["fold"].dtype, np.integer)
    assert not table.duplicated(["sample_id", "node_id"]).any()


def test_attribution_table_without_optional_graph_attributes():
    graphs = [_graph(index % 2, [index % 2, 1 - index % 2, index % 2], n_features=2) for index in range(4)]
    for graph in graphs:
        del graph.cluster
    classifier = nn.NodeClassifier(graphs, labels=np.arange(4) % 2, n_classes=2, hidden_dim=4, n_layers=1, device="cpu")
    classifier.fit(splits=[([0, 1], [2, 3]), ([2, 3], [0, 1])], epochs=1)

    table = classifier.attribution_table(1, n_steps=2)

    assert list(table.columns) == ATTRIBUTION_COLUMNS
    assert table["sample_id"].tolist() == [str(index) for index in range(4) for _ in range(3)]
    assert table["node_id"].tolist() == ["0", "1", "2"] * 4
    assert table["group_id"].isna().all()
    assert table[["x", "y", "cluster"]].isna().all().all()
    np.testing.assert_array_equal(table["fold"], np.repeat([2, 2, 1, 1], 3))


def test_attribution_table_indices_and_errors(four_class):
    classifier, graphs, labels, _ = four_class
    full = classifier.attribution_table(2, n_steps=2)
    subset = classifier.attribution_table(2, n_steps=2, indices=[9, 3])

    np.testing.assert_array_equal(subset["graph_index"], [9] * graphs[9].num_nodes + [3] * graphs[3].num_nodes)
    expected = pd.concat([full[full["graph_index"] == 9], full[full["graph_index"] == 3]], ignore_index=True)
    pd.testing.assert_frame_equal(subset, expected)

    empty = classifier.attribution_table(2, indices=[])
    assert list(empty.columns) == ATTRIBUTION_COLUMNS
    assert len(empty) == 0

    for bad in (4, -1, 1.0, True):
        with pytest.raises(ValueError, match=r"target_class must lie in range\(4\)\."):
            classifier.attribution_table(bad)
    for bad in (16, -1):
        with pytest.raises(IndexError, match=f"Graph index {bad} is out of range for 16 graphs"):
            classifier.attribution_table(0, indices=[0, bad])

    unfitted = nn.NodeClassifier(graphs, labels=labels, n_classes=4, device="cpu")
    with pytest.raises(RuntimeError, match=r"Call \.fit\(\) first\."):
        unfitted.attribution_table(0)


def test_attribution_table_skips_graphs_without_nodes():
    graphs = [_graph(index % 2, [index % 2, 1 - index % 2, index % 2], n_features=2, layout=True) for index in range(4)]
    classifier = nn.NodeClassifier(graphs, labels=np.arange(4) % 2, n_classes=2, hidden_dim=4, n_layers=1, device="cpu")
    classifier.fit(splits=[([0, 1], [2, 3]), ([2, 3], [0, 1])], epochs=1)
    empty = Data(x=torch.zeros((0, 2)), edge_index=torch.empty((2, 0), dtype=torch.long))
    graphs[1] = empty  # the classifier holds this list

    table = classifier.attribution_table(1, n_steps=2)

    assert sorted(set(table["graph_index"])) == [0, 2, 3]
    assert len(table) == 9
