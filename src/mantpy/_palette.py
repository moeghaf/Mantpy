"""Shared colour palette for consistent, accessible Mantpy figures.

Stable semantic anchors and colour-vision-deficiency-aware categorical
colours keep cells, ECM, conditions, and tissues consistent across plots.

Two design rules that fall out of this palette:

1. **Cells are always orange, ECM is always purple.** Whenever both appear
   in the same figure, they use the anchors below. ECM-only categorical
   plots (e.g. cluster IDs) can deviate, since the categorical axis is
   "cluster identity" not "cell vs ECM".

2. **Control/Disease is orthogonal to WT/KO.** A 2x2 design (Naive-WT,
   Naive-KO, Infected-WT, Infected-KO) uses two visual channels:
   green/vermillion for the group strip, and dark-grey/reddish-purple for
   the genotype fill. Never collapse them onto one channel.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Colour-vision-deficiency-aware categorical palettes for deutan, protan, and
# tritan accessibility and print-friendly output.
# ---------------------------------------------------------------------------

#: Okabe-Ito 4-distinct subset for two-factor (condition × genotype) figures
#: when both axes are encoded by hue.  Pairs that visually collide (the two
#: blues and the two oranges in the full Okabe-Ito set) are deliberately not
#: both present here.  For 2-factor designs where genotype is encoded on a
#: luminance axis instead, see :data:`GENOTYPE_LUMINANCE_PALETTE`.
OKABE_ITO_4DISTINCT: dict[str, str] = {
    "deep_blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "bluish_green": "#009E73",
}

#: Recommended pairing for two-factor designs: one experimental axis on hue,
#: the other on luminance. Use this with
#: :data:`CONDITION_HUE_PALETTE` so cluster swatches (which already occupy
#: the chromatic axes) never collide with the genotype channel.
GENOTYPE_LUMINANCE_PALETTE: dict[str, str] = {
    "dark": "#444444",
    "light": "#CCCCCC",
}
CONDITION_HUE_PALETTE: dict[str, str] = {
    "cool": "#0072B2",  # Okabe-Ito deep blue
    "warm": "#D55E00",  # Okabe-Ito vermillion
}

#: Categorical cluster palette for K ≤ 7 — hand-curated from Paul Tol's
#: "bright" qualitative palette and "discrete rainbow" anchors so that
#: (i) every cluster sits in a unique hue family, (ii) no cluster colour
#: collides with the four colours above, and (iii) ECM 5 stays in the cyan
#: family to preserve established cluster-colour consistency. Falls back
#: to a slightly more saturated
#: Trubetskoy palette for 8 ≤ K ≤ 12.
TOL_BRIGHT_7_CUSTOM: list[str] = [
    "#DC050C",  # 0 saturated red
    "#228833",  # 1 forest green
    "#4477AA",  # 2 medium blue
    "#DDAA33",  # 3 amber
    "#7A4794",  # 4 deep violet
    "#66CCEE",  # 5 cyan
    "#999933",  # 6 olive
]
TRUBETSKOY_12: list[str] = [
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9A6324",
]

#: Background-patch colour for reserved labels such as ``-1``.
CLUSTER_BG_COLOR: str = "#dddddd"

# ============================================================
# Anchor colours — use consistently across ALL figures
# ============================================================

CELL = "#E69F00"  # Okabe orange — cells, cellular nodes
ECM = "#7B5BA6"  # muted purple — ECM, matrix nodes, ECM compartments
# Okabe-Ito reddish-purple (#CC79A7) is a higher-saturation alternative.


# ============================================================
# Neutral greys — non-biological tiers, backgrounds, and text
# ============================================================

NEUTRAL_DARK = "#555555"
NEUTRAL_MID = "#9E9E9E"
NEUTRAL_LIGHT = "#C7C7C7"
NEUTRAL_LIGHTER = "#D9D9D9"
NEUTRAL_LIGHTEST = "#E0E0E0"


# ============================================================
# Condition / disease state
# ============================================================

CONTROL = "#009E73"  # Okabe bluish-green — Control / Naive / Healthy
DISEASE = "#D55E00"  # Okabe-Ito vermillion


# ============================================================
# Genotype — orthogonal axis to Control/Disease
# ============================================================

WT = "#4D4D4D"  # neutral dark grey
KO = "#CC79A7"  # Okabe reddish-purple


# ============================================================
# Small-intestine anatomical layers (Collagen-IV demonstration)
# ============================================================

# Post-hoc display colours for the five held-out anatomical reference layers.
# These labels and colours are never used during preprocessing, representation
# learning, graph construction, or clustering.
INTESTINE_LAYERS: dict[str, str] = {
    "Muscularis propria": "#D55E00",  # vermillion
    "Muscularis mucosae": "#E69F00",  # orange
    "Submucosa": "#009E73",  # bluish green
    "Crypts": "#0072B2",  # blue
    "Villus": "#CC79A7",  # reddish purple
}


# ============================================================
# ECM clusters
# ============================================================

# Re-use the first three colours for smaller cluster palettes so cluster IDs
# stay visually consistent between related analyses.
ECM_CLUSTERS: list[str] = [
    "#BDBDBD",  # ECM 0 — light grey (background-like / null cluster)
    "#0072B2",  # ECM 1 — blue
    "#56B4E9",  # ECM 2 — sky blue
    "#009E73",  # ECM 3 — bluish-green
    "#F0E442",  # ECM 4 — yellow (good for basement-membrane-like clusters)
    "#D55E00",  # ECM 5 — vermillion
    "#CC79A7",  # ECM 6 — reddish-purple
]


# ============================================================
# Canonical Okabe–Ito 8-colour palette (CVD-safe categorical)
# ============================================================
# The full Wong 2011 / Okabe & Ito 2008 qualitative set, named so figures can
# pull colour-vision-deficiency-safe categorical colours WITHOUT literal hex.
# All eight are mutually distinguishable under deuteranopia/protanopia and
# include black (no other named anchor exposes it).
OKABE_ITO: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}


# ============================================================
# Colormaps (continuous + diverging)
# ============================================================

SEQUENTIAL_INTENSITY = "magma"  # raw marker intensity (Col-IV, DAPI, etc.)
DIVERGING = "PuOr_r"  # log2 fold-changes, cosine similarity (centred at 0)
# use seaborn's `vlag` if you prefer the look of RdBu_r


# ============================================================
# Two-factor palette helper — maps factor values to a role palette
# ============================================================

#: Schemes supported by :func:`factor_palette`.  Each entry maps a scheme
#: name to ``(role_dict, ordered_role_keys)``.  The role keys define the
#: order in which factor values are assigned colours (the first value in
#: a call gets the first role colour, etc.). Add new schemes here as
#: additional two-factor designs are supported.
_FACTOR_SCHEMES: dict[str, tuple[dict[str, str], list[str]]] = {
    "condition_hue": (CONDITION_HUE_PALETTE, ["cool", "warm"]),
    "genotype_luminance": (GENOTYPE_LUMINANCE_PALETTE, ["dark", "light"]),
}


def factor_palette(
    values: list[str] | tuple[str, ...],
    *,
    scheme: str,
) -> dict[str, str]:
    """Map factor values to colours from a named two-tier palette scheme.

    Builds a two-factor colour mapping from a named scheme. Values are
    mapped to scheme roles by position: the first value gets the first
    role colour, the second value gets the second role colour, etc.

    Parameters
    ----------
    values
        Ordered list of factor values (e.g. ``['Naive', 'Infected']`` for
        a condition factor, ``['KO', 'WT']`` for a genotype factor).
        Order matters: it determines which role each value receives.
    scheme
        One of the keys of :data:`_FACTOR_SCHEMES`.  Currently:

        - ``"condition_hue"`` — Okabe-Ito blue (cool) + vermillion (warm).
          Use for nominal factors (Naive vs Infected, Healthy vs Diseased).
        - ``"genotype_luminance"`` — dark grey + light grey.  Use for the
          orthogonal factor in a 2×2 design (KO vs WT, etc.) so the
          chromatic axis stays free for clusters / tissues.

    Returns
    -------
    dict[str, str]
        Mapping of ``value → hex colour`` preserving the input order.

    Raises
    ------
    ValueError
        If ``scheme`` is unknown, or if ``len(values)`` exceeds the
        number of role colours available.

    Examples
    --------
    >>> import mantpy as mt
    >>> mt.palette.factor_palette(["Naive", "Infected"], scheme="condition_hue")
    {'Naive': '#0072B2', 'Infected': '#D55E00'}
    >>> mt.palette.factor_palette(["KO", "WT"], scheme="genotype_luminance")
    {'KO': '#444444', 'WT': '#CCCCCC'}

    See Also
    --------
    CONDITION_HUE_PALETTE : the raw role → hex dict for the hue axis.
    GENOTYPE_LUMINANCE_PALETTE : same for the luminance axis.
    """
    if scheme not in _FACTOR_SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}.  Choose from {list(_FACTOR_SCHEMES)}.")
    role_dict, roles = _FACTOR_SCHEMES[scheme]
    if len(values) > len(roles):
        raise ValueError(f"scheme {scheme!r} has only {len(roles)} roles ({roles}); got {len(values)} values.")
    return {v: role_dict[r] for v, r in zip(values, roles, strict=False)}


# Per-fold ROC curve colours for `classifier_roc(curves=...)`.  Pale blue
# per-fold lines + saturated mean overlay.
ROC_FOLD = "#7faedc"  # pale blue for the per-fold curves
ROC_MEAN = "#1f4e9b"  # saturated blue for the interpolated mean curve


# ============================================================
# AEC-ECM lung niche palette
# ============================================================
#
# Stable three-cluster palette for lung cell-ECM niche visualisations.
# ECM 0 preserves the shared ECM-purple anchor; ECM 1 and 2 use distinct
# teal and amber hues.

ECM_PALETTE_LUNG_PUBLISHED: dict[int, str] = {
    0: "#785EF0",  # ECM 0 — purple (matches the shared ECM anchor)
    1: "#2A9D8F",  # ECM 1 — teal (focal AEC-enriched niche cluster)
    2: "#E9A820",  # ECM 2 — yellow / amber
}

# Heterogeneous cell-ECM graph edge colours.
EDGE_CELL_CELL = "#1f4e9b"  # blue
EDGE_ECM_ECM = "#2d7a2d"  # green
EDGE_CELL_ECM = "#c64a8c"  # pink

# Cell-type palette overflow for the BALB/c lung CEG (12 cell types →
# extend the 12-colour Trubetskoy default with three darker anchors).
CELL_PALETTE_LUNG_EXT: list[str] = ["#7f0000", "#003f7f", "#3f3f3f"]

# Soft placeholder text for unavailable reconstructions.
PLACEHOLDER_GREY = "#888888"


__all__ = [
    "CELL",
    "ECM",
    "NEUTRAL_DARK",
    "NEUTRAL_MID",
    "NEUTRAL_LIGHT",
    "NEUTRAL_LIGHTER",
    "NEUTRAL_LIGHTEST",
    "CONTROL",
    "DISEASE",
    "WT",
    "KO",
    "INTESTINE_LAYERS",
    "ECM_CLUSTERS",
    "OKABE_ITO",
    "TOL_BRIGHT_7_CUSTOM",
    "TRUBETSKOY_12",
    "OKABE_ITO_4DISTINCT",
    "CLUSTER_BG_COLOR",
    "CONDITION_HUE_PALETTE",
    "GENOTYPE_LUMINANCE_PALETTE",
    "factor_palette",
    "SEQUENTIAL_INTENSITY",
    "DIVERGING",
    "ROC_FOLD",
    "ROC_MEAN",
    "ECM_PALETTE_LUNG_PUBLISHED",
    "EDGE_CELL_CELL",
    "EDGE_ECM_ECM",
    "EDGE_CELL_ECM",
    "CELL_PALETTE_LUNG_EXT",
    "PLACEHOLDER_GREY",
]
