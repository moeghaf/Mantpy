from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data
nn = pytest.importorskip("mantpy.nn")
ECMClusterGraphBundle = nn.ECMClusterGraphBundle
NodeClassifier = nn.NodeClassifier


def _graph(clusters):
    clusters = np.asarray(clusters, dtype=np.int64)
    graph = Data(
        x=torch.eye(2, dtype=torch.float32)[torch.from_numpy(clusters)],
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    graph.cluster = torch.from_numpy(clusters)
    graph.y = torch.tensor(0)
    return graph


def _fitted_classifier(graphs, probabilities, *, groups=None):
    if groups is None:
        labels = np.arange(len(graphs)) % 2
    else:
        group_labels = {
            group: index % 2
            for index, group in enumerate(dict.fromkeys(groups))
        }
        labels = np.asarray([group_labels[group] for group in groups])
    classifier = NodeClassifier(
        graphs,
        labels=labels,
        n_classes=2,
        device="cpu",
    )
    classifier._oof = probabilities
    classifier._groups = None if groups is None else np.asarray(groups)
    classifier._is_fit = True
    return classifier


def test_cluster_probability_table_uses_equal_roi_then_group_weights():
    graphs = [
        _graph([0, 0, 0, 0]),
        _graph([0, 1]),
        _graph([0, 0]),
        _graph([1]),
    ]
    classifier = _fitted_classifier(
        graphs,
        [
            np.array([[0.9, 0.1]] * 4),
            np.array([[0.1, 0.9], [0.7, 0.3]]),
            np.array([[0.3, 0.7], [0.3, 0.7]]),
            np.array([[0.2, 0.8]]),
        ],
        groups=["mouse_a", "mouse_a", "mouse_b", "mouse_b"],
    )

    table = classifier.cluster_probability_table()

    assert list(table.columns) == [
        "class_idx",
        "cluster",
        "unit_id",
        "mean_probability",
        "n_rois",
        "n_nodes",
    ]
    state0 = table.query("class_idx == 0 and cluster == 0")
    mouse_a = state0.query("unit_id == 'mouse_a'").iloc[0]
    assert mouse_a.mean_probability == pytest.approx((0.9 + 0.1) / 2)
    assert mouse_a.n_rois == 2
    assert mouse_a.n_nodes == 5
    mouse_b = state0.query("unit_id == 'mouse_b'").iloc[0]
    assert mouse_b.mean_probability == pytest.approx(0.3)
    assert mouse_b.n_rois == 1
    assert mouse_b.n_nodes == 2


def test_cluster_probability_summary_uses_equal_units_and_sample_sd():
    graphs = [
        _graph([0, 0, 0, 0]),
        _graph([0, 1]),
        _graph([0, 0]),
        _graph([1]),
    ]
    classifier = _fitted_classifier(
        graphs,
        [
            np.array([[0.9, 0.1]] * 4),
            np.array([[0.1, 0.9], [0.7, 0.3]]),
            np.array([[0.3, 0.7], [0.3, 0.7]]),
            np.array([[0.2, 0.8]]),
        ],
        groups=["mouse_a", "mouse_a", "mouse_b", "mouse_b"],
    )

    summary = classifier.cluster_probability_summary()

    row = summary.query("class_idx == 0 and cluster == 0").iloc[0]
    assert row.mean_probability == pytest.approx((0.5 + 0.3) / 2)
    assert row.unit_sd == pytest.approx(np.std([0.5, 0.3], ddof=1))
    assert row.n_units == 2
    assert row.n_rois == 3
    assert row.n_nodes == 7
    assert list(summary.columns) == [
        "class_idx",
        "cluster",
        "mean_probability",
        "unit_sd",
        "n_units",
        "n_rois",
        "n_nodes",
    ]


def test_cluster_probability_omits_missing_states_without_zero_imputation():
    graphs = [_graph([0]), _graph([0]), _graph([1])]
    classifier = _fitted_classifier(
        graphs,
        [
            np.array([[0.8, 0.2]]),
            np.array([[0.4, 0.6]]),
            np.array([[0.1, 0.9]]),
        ],
        groups=["mouse_a", "mouse_a", "mouse_b"],
    )

    table = classifier.cluster_probability_table()
    state1 = table.query("class_idx == 0 and cluster == 1")
    assert state1.unit_id.tolist() == ["mouse_b"]
    assert state1.iloc[0].mean_probability == pytest.approx(0.1)
    assert set(table.query("cluster == 1").class_idx) == {0, 1}

    summary = classifier.cluster_probability_summary()
    state1_summary = summary.query("class_idx == 0 and cluster == 1").iloc[0]
    assert state1_summary.mean_probability == pytest.approx(0.1)
    assert state1_summary.unit_sd == 0.0
    assert state1_summary.n_units == 1
    assert state1_summary.n_rois == 1
    assert state1_summary.n_nodes == 1


def test_cluster_probability_ungrouped_treats_each_graph_as_one_unit():
    graphs = [_graph([0, 0, 0]), _graph([0])]
    classifier = _fitted_classifier(
        graphs,
        [
            np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]),
            np.array([[0.1, 0.9]]),
        ],
    )

    table = classifier.cluster_probability_table()
    state0 = table.query("class_idx == 0 and cluster == 0")
    assert state0.unit_id.tolist() == ["0", "1"]
    assert state0.mean_probability.tolist() == pytest.approx([0.9, 0.1])
    assert state0.n_rois.tolist() == [1, 1]
    assert state0.n_nodes.tolist() == [3, 1]

    summary = classifier.cluster_probability_summary()
    row = summary.query("class_idx == 0 and cluster == 0").iloc[0]
    assert row.mean_probability == pytest.approx(0.5)
    assert row.unit_sd == pytest.approx(np.std([0.9, 0.1], ddof=1))
    assert row.n_units == 2
    assert row.n_rois == 2
    assert row.n_nodes == 4


def test_ecm_cluster_graph_bundle_has_an_informative_summary():
    bundle = ECMClusterGraphBundle(
        graphs=[object(), object()],
        labels=np.array([0, 1]),
        roi_ids=["roi_a", "roi_b"],
        class_names=["control", "disease"],
        cluster_ids=[0, 1, 2],
        feature_names=["ECM0", "ECM1", "ECM2"],
    )

    assert repr(bundle) == "\n".join(
        [
            "ECM graph-learning cohort",
            "  ROIs           2",
            "  classes        2",
            "  ECM states     3",
            "  node features  one-hot ECM state",
        ]
    )
