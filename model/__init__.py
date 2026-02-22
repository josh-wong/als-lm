"""GPT-2 model implementation for ALS-LM.

This package provides a clean standalone GPT-2 decoder-only transformer
with three named configurations (tiny, medium, 500M) for domain-specific
language model training on the ALS corpus.
"""

from model.model import GPT, GPTConfig, MODEL_CONFIGS

__all__ = ["GPT", "GPTConfig", "MODEL_CONFIGS"]
