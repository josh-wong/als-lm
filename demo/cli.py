"""Interactive CLI demo for querying ALS-LM models through Ollama.

This is a research artifact for exploring what a domain-specific language model
trained on ALS literature has learned. It is NOT a medical tool and must never
be used for medical decision-making. The model frequently generates
plausible-sounding but incorrect medical claims.

Usage::

    python -m demo.cli

Requires Ollama to be running with ALS-LM models registered. See the export
pipeline (``python -m export.export_pipeline``) to register models.
"""

import json
try:
    import readline  # noqa: F401 — enhances input() with history/editing
except ImportError:
    pass  # Unavailable on native Windows; input() still works without it
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Terminal styling (ANSI escape codes)
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()

BOLD = "\033[1m" if _IS_TTY else ""
DIM = "\033[2m" if _IS_TTY else ""
YELLOW = "\033[33m" if _IS_TTY else ""
CYAN = "\033[36m" if _IS_TTY else ""
GREEN = "\033[32m" if _IS_TTY else ""
RED = "\033[31m" if _IS_TTY else ""
RESET = "\033[0m" if _IS_TTY else ""

OLLAMA_BASE = "http://localhost:11434"

# ---------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------


def _ollama_reachable() -> bool:
    """Return True if the Ollama API is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        return resp.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def _list_als_models() -> List[Dict[str, Any]]:
    """Fetch models from Ollama and filter to ALS-LM models.

    Returns a list of dicts with keys ``name`` and ``size`` (bytes).
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return []

    models = resp.json().get("models", [])
    als_models = []
    for m in models:
        name = m.get("name", "")
        if name.startswith("als-lm-"):
            als_models.append({
                "name": name,
                "size": m.get("size", 0),
            })

    # Sort by name so order is deterministic
    als_models.sort(key=lambda m: m["name"])
    return als_models


def _format_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string (MB or GB)."""
    if size_bytes <= 0:
        return "? MB"
    mb = size_bytes / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def _select_model(models: List[Dict[str, Any]]) -> Optional[str]:
    """Display a numbered menu and return the selected model name."""
    print(f"\n{BOLD}Available ALS-LM models:{RESET}\n")

    # Pick a single default: prefer als-lm-500m:q8_0, then first :q8_0, then first model
    default_idx = 0
    for i, m in enumerate(models):
        if m["name"] == "als-lm-500m:q8_0":
            default_idx = i
            break
        if ":q8_0" in m["name"] and default_idx == 0:
            default_idx = i

    for i, m in enumerate(models):
        name = m["name"]
        size_str = _format_size(m["size"])
        tag = "  [default]" if i == default_idx else ""
        print(f"  {i + 1}. {name:<25} ({size_str}){tag}")

    print()
    try:
        choice = input(f"Select model [{default_idx + 1}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return models[default_idx]["name"]

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]["name"]
    except ValueError:
        pass

    print(f"{RED}Invalid selection.{RESET}")
    return None


# ---------------------------------------------------------------------------
# Medical safety disclaimer
# ---------------------------------------------------------------------------


_DISCLAIMER = f"""
{YELLOW}{BOLD}==============================================================
                    IMPORTANT DISCLAIMER
=============================================================={RESET}

{YELLOW}This is a research artifact, NOT a medical tool.

ALS-LM is a small language model (500M parameters) trained on
public ALS literature as a machine learning research project.
It frequently generates plausible-sounding but factually
incorrect medical claims.

Do NOT use this model for medical decision-making.
Always consult qualified healthcare professionals for any
medical questions or concerns.

By proceeding, you acknowledge that you understand these
limitations and will not rely on the model's output for
medical purposes.{RESET}
"""


def _show_disclaimer() -> bool:
    """Display the medical safety disclaimer and require acknowledgment.

    Returns True if the user accepts, False otherwise.
    """
    print(_DISCLAIMER)
    try:
        response = input(
            f"Do you understand and accept these terms? ({GREEN}yes{RESET}/{RED}no{RESET}): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return response in ("yes", "y")


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


def _generate(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Optional[Tuple[int, float]]:
    """Send a prompt to Ollama and stream the response token by token.

    Returns (token_count, tokens_per_second) on success, or None on error.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}Connection lost. Is Ollama still running?{RESET}")
        return None
    except requests.exceptions.Timeout:
        print(f"\n{RED}Request timed out (120s). Try a shorter prompt or lower max_tokens.{RESET}")
        return None
    except requests.exceptions.HTTPError as exc:
        print(f"\n{RED}Ollama error: {exc}{RESET}")
        return None

    eval_count = 0
    eval_duration_ns = 0

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        # Print token text immediately
        token_text = chunk.get("response", "")
        if token_text:
            print(token_text, end="", flush=True)

        if chunk.get("done", False):
            eval_count = chunk.get("eval_count", 0)
            eval_duration_ns = chunk.get("eval_duration", 0)

    # Newline after streamed output
    print()

    if eval_count > 0 and eval_duration_ns > 0:
        tok_per_sec = eval_count / (eval_duration_ns / 1e9)
        return (eval_count, tok_per_sec)

    return (0, 0.0)


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class Session:
    """Holds mutable session state for the REPL."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 256
        self.history: List[str] = []


def _handle_slash_command(cmd: str, session: Session) -> bool:
    """Handle a slash command. Returns True if the REPL should exit."""
    parts = cmd.split(None, 1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/quit", "/exit"):
        return True

    if command == "/help":
        print(f"""
{BOLD}Available commands:{RESET}

  /help                Show this help message
  /info                Show current model and settings
  /temperature <val>   Set temperature (0.0-2.0, current: {session.temperature})
  /max_tokens <val>    Set max tokens (1-2048, current: {session.max_tokens})
  /history             Show last 10 prompts from this session
  /clear               Clear the terminal screen
  /quit, /exit         Exit the CLI
""")
        return False

    if command == "/info":
        print(f"\n  {BOLD}Model:{RESET}       {session.model}")
        print(f"  {BOLD}Temperature:{RESET} {session.temperature}")
        print(f"  {BOLD}Max tokens:{RESET}  {session.max_tokens}")
        print()
        return False

    if command == "/temperature":
        if not arg:
            print(f"{RED}Usage: /temperature <value> (0.0-2.0){RESET}")
            return False
        try:
            val = float(arg)
            if not 0.0 <= val <= 2.0:
                raise ValueError
            session.temperature = val
            print(f"{GREEN}Temperature set to {val}{RESET}")
        except ValueError:
            print(f"{RED}Invalid value. Temperature must be a number between 0.0 and 2.0.{RESET}")
        return False

    if command == "/max_tokens":
        if not arg:
            print(f"{RED}Usage: /max_tokens <value> (1-2048){RESET}")
            return False
        try:
            val = int(arg)
            if not 1 <= val <= 2048:
                raise ValueError
            session.max_tokens = val
            print(f"{GREEN}Max tokens set to {val}{RESET}")
        except ValueError:
            print(f"{RED}Invalid value. Max tokens must be an integer between 1 and 2048.{RESET}")
        return False

    if command == "/history":
        if not session.history:
            print(f"{DIM}No prompts in this session yet.{RESET}")
            return False
        recent = session.history[-10:]
        print(f"\n{BOLD}Recent prompts:{RESET}\n")
        for i, prompt in enumerate(recent, 1):
            # Truncate long prompts for display
            display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
            print(f"  {i}. {display}")
        print()
        return False

    if command == "/clear":
        print("\033[2J\033[H", end="", flush=True)
        return False

    print(f"{RED}Unknown command: {command}{RESET}")
    print(f"Type {BOLD}/help{RESET} to see available commands.")
    return False


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ALS-LM interactive CLI demo."""

    # --- Ollama connection check ---
    if not _ollama_reachable():
        print(f"{RED}Cannot connect to Ollama at {OLLAMA_BASE}.{RESET}")
        print("Make sure Ollama is running. You can start it with:")
        print(f"  {BOLD}ollama serve{RESET}")
        sys.exit(1)

    # --- Model discovery ---
    models = _list_als_models()
    if not models:
        print(f"{RED}No ALS-LM models found in Ollama.{RESET}")
        print("Register models by running the export pipeline:")
        print(f"  {BOLD}python -m export.export_pipeline{RESET}")
        sys.exit(1)

    # --- Model selection ---
    model = _select_model(models)
    if model is None:
        print("No model selected. Exiting.")
        sys.exit(0)

    print(f"\nUsing model: {BOLD}{model}{RESET}")

    # --- Medical safety disclaimer ---
    if not _show_disclaimer():
        print("Disclaimer not accepted. Exiting.")
        sys.exit(0)

    # --- REPL loop ---
    session = Session(model)
    prompt_str = f"{CYAN}[{model}]{RESET} > "

    print(f"\n{BOLD}ALS-LM Interactive Demo{RESET}")
    print(f"Type a prompt to generate text. Type {BOLD}/help{RESET} for commands.\n")

    try:
        while True:
            try:
                user_input = input(prompt_str).strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                should_exit = _handle_slash_command(user_input, session)
                if should_exit:
                    break
                continue

            # Record prompt in session history
            session.history.append(user_input)

            # Generate response
            result = _generate(
                session.model,
                user_input,
                session.temperature,
                session.max_tokens,
            )

            if result is not None:
                token_count, tok_per_sec = result
                if token_count > 0:
                    print(f"{DIM}[{token_count} tokens | {tok_per_sec:.1f} tok/s]{RESET}")
                print()

    except KeyboardInterrupt:
        print()

    print(f"\n{DIM}Goodbye.{RESET}")


if __name__ == "__main__":
    main()
