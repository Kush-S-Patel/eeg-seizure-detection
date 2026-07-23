from __future__ import annotations

import sys


def get_int_max_str_digits() -> int:
    return 4300


def set_int_max_str_digits(maxdigits: int) -> None:
    return None


sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]
print("patched", hasattr(sys, "get_int_max_str_digits"))

import torch

m = torch.nn.Linear(2, 1)
torch.optim.AdamW(m.parameters())
print("adamw ok", torch.__version__)
