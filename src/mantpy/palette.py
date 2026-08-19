"""Public colour palettes and factor-aware palette helpers."""

from mantpy import _palette as _impl

__all__ = _impl.__all__

for _name in __all__:
    globals()[_name] = getattr(_impl, _name)

del _name
