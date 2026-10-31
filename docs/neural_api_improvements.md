# Reusable neural analysis

`PatchEncoder` trains a shared image encoder with PyTorch Lightning. Existing
calls retain the original architecture, augmentation, optimizer and output
scaling. Supply a PyTorch module or a factory to change the backbone; factories
are invoked after seeding. Projection-head outputs enter contrastive training,
while the public embedding always comes from the backbone.

```python
encoder = mt.nn.PatchEncoder(
    backbone=make_cnn,
    backbone_kwargs={"channels": 6, "latent_dim": 96},
    latent_dim=96,
    projection=make_projection,
    augment=augment_patches,
)
encoder.fit(patches, max_epochs=100, random_state=0, callbacks=callbacks)
encoder.transform_adata(adata, standardize=False)
raw = encoder.transform(patches, standardize=False)
```

The backbone accepts `(N, C, H, W)` and returns `(N, latent_dim)`. Augmentations
preserve the patch shape. `optimizer_factory`, `optimizer_kwargs`,
`scheduler_factory` and `scheduler_kwargs` customize training without replacing
the loop. `scheduler=None` disables the default cosine schedule. `drop_last=None`
retains the original one-patch-tail rule; pass `False` when a custom model
supports that tail. `history` contains detached epoch losses and
`optimizer_updates` counts actual Lightning optimizer steps.

For parity with a historical manual DataLoader loop, use
`PatchEncoder.fit(..., exhaust_sampler=True)`. It consumes complete shuffled
epoch draws even when Lightning stops after one known batch. The default
retains existing Lightning sampling behavior. Batch contents and order are
unchanged; the option controls the dedicated sampler's random stream across
epochs.

Trainer options pass through `fit`, including `deterministic="warn"` for a
custom CUDA model whose operations lack deterministic implementations. This
permits training with warnings; it does not guarantee identical repeated GPU
fits. The liver tutorial uses this option for adaptive maximum pooling.

`save` stores weights, configuration and the fitted feature scaler. It does not
pickle a custom model. Reload a custom architecture explicitly:

```python
restored = mt.nn.PatchEncoder.load(path, backbone=make_cnn)
```

`GraphMAE` retains its original defaults and exposes the fitted module,
detached `history["train_loss"]`, `optimizer_updates` and an optional epoch
callback. Optimizer/scheduler factories and reconstruction-loss callables can
be supplied to `train` or `encode_graphmae`. Raw output is explicit:

```python
latent = model.get_node_latent(standardize_output=False)
```

For a new graph, the legacy default standardizes that graph's input and output
independently. Select `feature_transform="fitted"` and
`output_transform="fitted"` to reuse the training graph's feature and latent
scales. `encode_graphmae(include_edge_features=False)` omits edge features
explicitly; its default remains `True`.

`NodeClassifier` keeps graphs and labels in its constructor, and seeds/splits
in `fit`. Biological groups can accompany explicit splits; training and test
groups must be disjoint, but one held-out group need not contain every class.

```python
classifier = mt.nn.NodeClassifier(
    graphs, labels=labels, model_factory=make_graphsage,
)
classifier.fit(
    groups=mouse_ids, splits=splits, seeds=[0, 1, 2],
    loss_reduction="roi", reset_seed_per_fit=True,
)
```

`loss_reduction="node"` preserves the original pooled-node loss. `"roi"`
averages losses within ROIs before averaging across ROIs. Class weights can be
training-derived (`"balanced"`), a supplied vector, or disabled with `None`.
For ROI loss, `roi_class_weight_mode="after"` applies weights after each ROI
mean; `"node"` applies them inside cross-entropy. The difference matters with
label smoothing. `reset_seed_per_fit=False` retains the historical continuous
random stream across folds; `True` resets each independent fit.

For custom experiment orchestration, `train_node_classifier` fits one model
using only the supplied training graphs. It returns a `GraphClassifierFit`
with the fitted model, training-only mean/scale, detached history and
`predict_proba`. Splitting and checkpoint locations remain caller-controlled.
`NodeClassifier` uses this same engine.

Integrated gradients attribute the mean requested class logit to the features.
The default baseline is zero in training-standardized space. The tuple return
remains compatible; `return_details=True` adds signed arrays, completeness
deltas, node identifiers and, for an ensemble, per-seed signed attributions.
Signed seed attributions are averaged before computing their absolute spatial
magnitude. The standalone `node_integrated_gradients` supports saved models and
fitted transforms without refitting.

`NeighbourCompositionBaseline(estimator=...)` clones a compatible scikit-learn
classifier instead of always using logistic regression. `transform` exposes
its fitted-vocabulary neighborhood proportions with stable column names and
the original patch-table index.
