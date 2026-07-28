# Public datasets

Mantpy provides one-line access to the four datasets used in its public
tutorials:

```python
import mantpy as mt

intestine = mt.datasets.coliv_intestine()
lung = mt.datasets.balbc_pbs_lung()
liver = mt.datasets.schistosoma_ecm()
prostate = mt.datasets.prostate_he_visium()
```

The data are not bundled in the Python wheel. Each call downloads one archive
from the immutable [Zenodo version record](https://doi.org/10.5281/zenodo.21538382),
verifies the archive size and SHA-256, extracts it through a data-only safety
boundary, and verifies every inner file against `manifest.json`. Downloads are
anonymous and reusable offline after the first successful call.

## Cache location

The cache is selected in this order:

1. `cache_dir=...` passed to the loader;
2. the `MANTPY_CACHE` environment variable;
3. `~/.cache/mantpy`.

For example:

```python
data = mt.datasets.coliv_intestine(cache_dir="D:/mantpy-data")
```

Each returned bundle has:

- `paths`: verified files, grouped by manifest role;
- `provenance`: portable public source and licence metadata;
- `quickstart`: verified optional intermediates as `pathlib.Path` values.

Quick-start files are never silently substituted for raw data. Tutorials choose
whether to use them, validate their scientific metadata, and otherwise run the
full workflow from the raw inputs.

The prostate loader obtains the human Matrisome masterlist from the
authoritative Matrisome Project export. Mantpy pins its expected SHA-256, so an
upstream revision cannot silently change the tutorial gene universe.

## Local verified bundles

For testing or an institutional mirror, pass `source_dir=...`. The directory
must follow the same manifest contract and receives the same complete size and
SHA-256 validation as an extracted download. Pickle and executable files are
not accepted.

The manifest schema is version 1. It contains `dataset`, `license`, optional
portable `provenance` and `metadata` objects, plus a `files` list. Every file
entry contains `path`, `role`, `size`, and `sha256`; repeated per-sample roles
also carry `sample_id`. Optional intermediates set `quickstart: true` and may
provide a stable `key`.
