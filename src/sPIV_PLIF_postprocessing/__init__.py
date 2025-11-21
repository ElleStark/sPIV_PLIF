"""sPIV_PLIF_postprocessing package

Lightweight package initializer that exposes package metadata and
provides a convenient import surface for commonly-used utilities.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

try:
    # expose top-level helpers if present (silently ignore missing)
    from . import analysis  # type: ignore
    __all__.append("analysis")
except Exception:
    # module may not exist yet; don't raise on import
    pass
