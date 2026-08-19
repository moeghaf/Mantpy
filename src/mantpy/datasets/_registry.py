"""Pinned metadata for Mantpy's public tutorial data record."""

from __future__ import annotations

from typing import Final

# Immutable metadata for the pinned Zenodo version record.  Loaders verify
# every downloaded byte against these sizes and SHA-256 digests.
DATASET_REGISTRY: Final[dict[str, object]] = {
    "schema_version": 1,
    "record_id": 21538382,
    "base_url": "https://zenodo.org",
    "license": "CC-BY-4.0",
    "manifest": {
        "filename": "mantpy-datasets-manifest.json",
        "size": 862,
        "sha256": "b36ece373446209d165d4e7bbe6a3f45e0588fce99a1ae91372ca62e2850b4ce",
    },
    "datasets": {
        "coliv_intestine": {
            "filename": "mantpy-coliv-intestine.tar.gz",
            "size": 93408111,
            "sha256": "02d14a2f8a758dd7103ca9c86fb3377a85d3738e957f263dc0ce8c61bc774414",
        },
        "balbc_pbs_lung": {
            "filename": "mantpy-balbc-pbs-lung.tar.gz",
            "size": 71092055,
            "sha256": "28623cc27d2b38f2c6d9397b512f5e79c9b7b6bd51acd13258188da01cfd6f28",
        },
        "schistosoma_ecm": {
            "filename": "mantpy-schistosoma-ecm.tar.gz",
            "size": 37610693,
            "sha256": "bf6f8010f3a37d3af77863502328a525f82ef791a8e92f0ba51222c2d0b0ebf1",
        },
        # Part of the immutable Zenodo deposit (and its pinned root
        # manifest); not exposed through a public loader since 1.0.0.
        "prostate_he_visium": {
            "filename": "mantpy-prostate-he-visium.tar.gz",
            "size": 193646345,
            "sha256": "679a996e412ce9b9b119f5a96d83013fa5298cea5e2f4139c825aafbe1cb15ef",
        },
    },
}
