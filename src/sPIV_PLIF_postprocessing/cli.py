"""Small CLI wrapper so the pipeline is available as an entry point.

This module is intentionally tiny: it parses command-line args, loads a
TOML config, and delegates to `pipeline.run`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import tomllib
from typing import Any, Dict

from .pipeline import run as run_pipeline


logger = logging.getLogger("sPIV_PLIF.cli")


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("rb") as fh:
        return tomllib.load(fh)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="postprocess")
    p.add_argument("-c", "--config", required=True, help="Path to TOML config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    cfg = load_config(args.config)
    if args.dry_run:
        logger.info("Dry run - config validated: %s", args.config)
        return
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
