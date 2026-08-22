# Advanced: the full analyses

The tutorials in this section run on a synthetic ROI so they execute in
seconds. The analyses behind the
[Mantpy preprint](https://doi.org/10.1101/2025.06.04.657781) run on real cohorts, take
minutes to hours, and download hundreds of megabytes — so they live in the
[reproducibility repository](https://github.com/moeghaf/mantpy_reproducibility)
rather than here; the intestine and lung workflows open directly in Colab.

Every dataset comes from a single immutable, checksummed
[Zenodo record](https://doi.org/10.5281/zenodo.21538382) (CC BY 4.0).

## The three workflows

:::{list-table}
:header-rows: 1
:widths: 26 30 14 30

* - Workflow
  - What it shows
  - Download
  - Open
* - **Intestine (Collagen-IV)**
  - Self-supervised ECM representations with GraphMAE, benchmarked against
    published comparators
  - 93 MB
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/coliv_intestine_graphmae.ipynb)
* - **Mouse lung**
  - Joint cell–ECM graphs across a cohort, with enrichment and niche structure
  - 71 MB
  - [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moeghaf/mantpy_reproducibility/blob/main/notebooks/joint_lung_cell_ecm.ipynb)
* - **Schistosoma liver**
  - Multi-marker ECM states around granulomas, with a graph classifier and
    explainability
  - 38 MB
  - [Executed notebook](https://github.com/moeghaf/mantpy_reproducibility/blob/main/notebooks/schistosoma_ecm_liver.ipynb) (local run)
:::

## Fast mode and full mode

Each notebook runs in one of two modes, and the difference is not cosmetic.

**Fast mode** is what Colab gets by default. Permutation counts are reduced,
model training is shortened, and heavy comparators are skipped. The analysis
runs end to end and the figures are recognisably the published ones, but the
numbers are not the published numbers.

**Full mode** reproduces the manuscript exactly, and is correspondingly
demanding: a Linux/amd64 host, a pinned dependency lockfile, and — for the
intestine workflow — Docker, to build the comparator environments. The intestine and lung
workflows additionally refuse to start unless the environment matches the frozen
record, which is deliberate: a bit-exact claim is worth nothing if the
environment silently drifts.

Colab cannot satisfy those requirements, so full mode is a prepared-host
activity. The reproducibility repository's
[README](https://github.com/moeghaf/mantpy_reproducibility#reproducing-the-public-workflows)
documents what that host needs.

Both modes are selectable:

```bash
MANTPY_TUTORIAL_MODE=fast   # or full
```

## Which one to read first

If you are new to Mantpy, the mouse lung workflow is the most direct
continuation of these tutorials — it is the cell–ECM graph you have already
built, applied to a real cohort.
