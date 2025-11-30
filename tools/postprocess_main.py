"""Thin dataset runner wrapper kept in `tools/`.

This file intentionally stays small: it loads a TOML config and calls
the package-level `pipeline.run(cfg)` function. Keep dataset-specific
ad-hoc scripts in `tools/` so the package (`src/...`) contains the
reusable logic.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import tomlkit
from typing import Any, Dict


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("sPIV_PLIF.postprocess")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("rb") as fh:
        return tomlkit.load(fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="postprocess_main")
    parser.add_argument("-c", "--config", required=True, help="Path to TOML config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO

    logger = setup_logging(level)
    cfg = load_config(args.config)

    # import pipeline from package; keep import local so tools/ can be used
    # without installing the package (useful during development)
    try:
        from sPIV_PLIF_postprocessing.pipeline import run as run_pipeline
    except Exception:
        # fallback to old-style utils script if package not installed
        logger.debug("Package import failed, attempting local import of pipeline module")
        import importlib

        pipeline_mod = importlib.import_module("sPIV_PLIF_postprocessing.pipeline")
        run_pipeline = getattr(pipeline_mod, "run")

    if args.dry_run:
        logger.info("Dry run - configuration loaded: %s", args.config)
        return

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
