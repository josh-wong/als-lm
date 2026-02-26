"""Shared utilities for the ALS-LM evaluation package.

Provides project root discovery and default path resolution so that eval
scripts work correctly regardless of the caller's working directory. When
scripts are invoked from project root, the defaults match the original
relative paths. When invoked from elsewhere, find_project_root() walks up
the directory tree to locate the project and returns absolute paths.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_project_root():
    """Locate the project root by walking up from this file's directory.

    Looks for a directory containing an ``eval/`` subdirectory with a
    ``questions.json`` file (the benchmark). Starts from the directory
    containing this module and walks toward the filesystem root.

    Returns:
        A ``pathlib.Path`` pointing to the project root.

    Raises:
        SystemExit: If the project root cannot be determined. The error
            message suggests passing explicit ``--benchmark`` and
            ``--registry`` flags as a workaround.
    """
    # Start from the directory containing this file (eval/)
    start = Path(__file__).resolve().parent

    # The eval/ directory itself lives inside the project root, so go up one
    candidate = start.parent
    if (candidate / "eval").is_dir():
        logger.info("Project root: %s", candidate)
        return candidate

    # Walk upward as a fallback (handles unusual layouts)
    current = start
    while current != current.parent:
        current = current.parent
        if (current / "eval").is_dir():
            logger.info("Project root: %s", current)
            return current

    raise SystemExit(
        "Could not determine project root (no parent directory contains "
        "an eval/ subdirectory). Pass explicit --benchmark and --registry "
        "flags to specify file locations."
    )


def resolve_default_paths(project_root):
    """Return default benchmark and registry paths for a given project root.

    Args:
        project_root: A ``pathlib.Path`` pointing to the project root
            directory.

    Returns:
        A dict with keys ``benchmark`` and ``registry``, each mapping to
        an absolute ``pathlib.Path``.
    """
    return {
        "benchmark": project_root / "eval" / "questions.json",
        "registry": project_root / "eval" / "entity_registry.json",
    }
