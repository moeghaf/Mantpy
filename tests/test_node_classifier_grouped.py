from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data
nn = pytest.importorskip("mantpy.nn")
node_classifier = pytest.importorskip("mantpy.nn._node_classifier")


def _graph(label: int, clusters: list[int], n_features: int = 4):
    cluster_array = np.asarray(clusters, dtype=np.int64)
    graph = Data(
        x=torch.eye(n_features, dtype=torch.float32)[
            torch.from_numpy(cluster_array)
        ],
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    graph.cluster = torch.from_numpy(cluster_array)
    graph.y = torch.tensor(label)
    return graph


def _four_class_cohort():
    graphs = []
    labels = []
    groups = []
    for label in range(4):
        for mouse in range(2):
            group = f"class{label}_mouse{mouse}"
            for roi in range(2):
                other = (label + mouse + roi + 1) % 4
                graphs.append(_graph(label, [label, label, label, other]))
                labels.append(label)
                groups.append(group)
    return graphs, np.asarray(labels), np.asarray(groups)


def test_internal_stratified_group_folds_hold_one_mouse_per_class():
    _, labels, groups = _four_class_cohort()

    folds = node_classifier._stratified_group_folds(labels, groups, n_splits=2)

    assert len(folds) == 2
    seen = np.zeros(len(labels), dtype=int)
    for train_indices, test_indices in folds:
        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])
        assert train_groups.isdisjoint(test_groups)
        assert len(test_groups) == 4
        assert set(labels[test_indices]) == {0, 1, 2, 3}
        assert all(np.sum(groups[test_indices] == group) == 2 for group in test_groups)
        seen[test_indices] += 1
    np.testing.assert_array_equal(seen, 1)


def test_grouped_fit_uses_train_fold_transforms_and_reports_eight_mice():
    graphs, labels, groups = _four_class_cohort()
    classifier = nn.NodeClassifier(
        graphs,
        labels=labels,
        n_classes=4,
        hidden_dim=4,
        n_layers=1,
        dropout=0.0,
        device="cpu",
    )

    classifier.fit(
        n_splits=2,
        groups=groups,
        seeds=[0, 1],
        epochs=1,
    )

    assignments = classifier.split_assignments()
    assert assignments["group_id"].nunique() == 8
    assert set(assignments["fold"]) == {1, 2}
    for fold in (1, 2):
        test_groups = set(assignments.loc[assignments["fold"] == fold, "group_id"])
        train_groups = set(assignments.loc[assignments["fold"] != fold, "group_id"])
        assert test_groups.isdisjoint(train_groups)

    transforms = classifier.fold_transform_table()
    assert set(transforms["seed"]) == {0, 1}
    assert set(transforms["fold"]) == {1, 2}
    for fold in (1, 2):
        train_indices = assignments.index[assignments["fold"] != fold].to_numpy()
        train_x = torch.cat([graphs[index].x for index in train_indices]).numpy()
        expected_mean = train_x.mean(axis=0)
        expected_sd = train_x.std(axis=0)
        expected_sd[expected_sd == 0] = 1.0
        observed = (
            transforms.query("seed == 0 and fold == @fold")
            .sort_values("feature")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(observed["train_mean"], expected_mean)
        np.testing.assert_allclose(observed["train_sd"], expected_sd)

    metrics = classifier.cv_metrics()
    assert len(metrics["group_ids"]) == 8
    assert metrics["group_confusion"].shape == (4, 4)
    assert len(classifier.seed_metrics()) == 2
    prediction_table = classifier.prediction_table(level="group")
    assert set(prediction_table["estimate"]) == {"ensemble", "seed"}
    assert prediction_table.query("estimate == 'ensemble'")["unit_id"].nunique() == 8
    assert len(prediction_table.query("estimate == 'seed'")) == 16


def test_grouped_multiseed_ig_uses_every_held_out_fit():
    graphs, labels, groups = _four_class_cohort()
    classifier = nn.NodeClassifier(
        graphs,
        labels=labels,
        n_classes=4,
        hidden_dim=4,
        n_layers=1,
        dropout=0.0,
        device="cpu",
    ).fit(n_splits=2, groups=groups, seeds=[0, 1], epochs=1)
    call_counts = [0, 0]
    handles = []
    for index, fit in enumerate(classifier._held_out_fits[0]):
        def _count_call(_module, _inputs, _output, *, model_index=index):
            call_counts[model_index] += 1

        handles.append(fit.net.register_forward_hook(_count_call))
    try:
        node_importance, feature_attribution = classifier.integrated_gradients(
            0,
            target_class=0,
            n_steps=3,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert all(count > 0 for count in call_counts)
    assert node_importance.shape == (graphs[0].num_nodes,)
    assert feature_attribution.shape == graphs[0].x.shape
    assert np.isfinite(node_importance).all()
    assert np.isfinite(feature_attribution).all()
