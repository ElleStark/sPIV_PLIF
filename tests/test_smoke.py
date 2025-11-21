import sys
from pathlib import Path


def test_cli_dry_run():
    # Ensure package import from local src/ during test
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from sPIV_PLIF_postprocessing import cli

    config = repo_root / "configs" / "example.toml"
    # Should not raise; runs in dry-run mode
    cli.main(["-c", str(config), "--dry-run"])
