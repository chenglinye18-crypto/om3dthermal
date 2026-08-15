"""OM3D Thermal geometry front-end."""

from .config import (
    OrthogonalM3DTemplateConfig,
    SimulationConfig,
    UnresolvedPhysicalParametersError,
    load_config,
    load_orthogonal_m3d_template,
)

__all__ = [
    "OrthogonalM3DTemplateConfig",
    "SimulationConfig",
    "UnresolvedPhysicalParametersError",
    "load_config",
    "load_orthogonal_m3d_template",
]
__version__ = "0.1.0"


def _git_metadata(repo_or_path) -> dict:
    """Collect the current main-repo + DreamRAM commit/branch for metadata.

    The function is best-effort: any git failure (e.g. detached head,
    missing git binary) returns a partial dict with ``None`` for the
    missing fields, never raises. ``repo_or_path`` is a directory that
    contains a ``.git`` (main repo) plus a ``third_party/DreamRAM`` peer.
    """
    import subprocess
    from pathlib import Path
    root = Path(repo_or_path).resolve()
    # walk up to find the .git boundary
    cur = root if (root / ".git").exists() else root.parent
    for ancestor in (cur, *cur.parents):
        if (ancestor / ".git").exists():
            cur = ancestor
            break
    meta: dict = {"main_repo_commit": None, "main_repo_branch": None,
                  "dreamram_commit": None, "dreamram_branch": None}
    try:
        meta["main_repo_commit"] = subprocess.check_output(
            ["git", "-C", str(cur), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        meta["main_repo_branch"] = subprocess.check_output(
            ["git", "-C", str(cur), "branch", "--show-current"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    dr = cur / "third_party" / "DreamRAM"
    if dr.exists():
        try:
            meta["dreamram_commit"] = subprocess.check_output(
                ["git", "-C", str(dr), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
            meta["dreamram_branch"] = subprocess.check_output(
                ["git", "-C", str(dr), "branch", "--show-current"],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return meta
