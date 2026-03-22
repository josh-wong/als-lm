#!/usr/bin/env python3
"""Shared console and utility functions for QLoRA pipeline scripts."""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared paths (single source of truth for project root and config location)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "qlora.json"


# Default assistant tag for Qwen chat template format. Used as fallback
# when the tokenizer is not available (e.g., in tests without GPU packages).
# The actual tag is derived dynamically in train_qlora._derive_assistant_tag().
DEFAULT_ASSISTANT_TAG = "<|im_start|>assistant\n"


def load_qlora_config() -> dict:
    """Load and return the QLoRA config from configs/qlora.json."""
    if not CONFIG_PATH.exists():
        fatal(f"Config not found at {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return json.load(f)

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def section(title: str):
    """Print a section header with separators."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def status(msg: str):
    """Print a status message."""
    print(f"  {msg}")


def ok(msg: str):
    """Print a green success message."""
    print(f"  {GREEN}[OK]{RESET} {msg}")


def warn(msg: str):
    """Print a yellow warning message."""
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def fatal(msg: str):
    """Print a red fatal error and exit."""
    print(f"\n  {RED}FATAL:{RESET} {msg}", file=sys.stderr)
    sys.exit(1)


def print_pass(msg: str):
    """Print a green PASS message."""
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def print_fail(msg: str):
    """Print a red FAIL message."""
    print(f"  {RED}[FAIL]{RESET} {msg}")
