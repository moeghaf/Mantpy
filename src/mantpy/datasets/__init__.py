"""One-line access to the public datasets used by Mantpy tutorials.

The data live in an immutable, checksummed Zenodo record and are cached outside
the package.  Every loader returns a :class:`mantpy.ds.Bunch`, so fields support
both attribute and dictionary access.

:func:`toy_ecm_roi` is the exception — it synthesises a small ROI in memory, so
documentation and quick experiments need no download at all.
"""

from mantpy.datasets._loaders import (
    balbc_pbs_lung,
    coliv_intestine,
    prostate_he_visium,
    schistosoma_ecm,
)
from mantpy.datasets._toy import toy_ecm_roi

__all__ = [
    "balbc_pbs_lung",
    "coliv_intestine",
    "prostate_he_visium",
    "schistosoma_ecm",
    "toy_ecm_roi",
]
