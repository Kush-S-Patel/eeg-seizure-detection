"""Weakly supervised Neurotech EEG seizure-detector starter."""

from __future__ import annotations

import sys

# Python 3.11.0rc1 (some Ubuntu images) lacks these; modern PyTorch imports them.
# Signatures must match CPython so torch._dynamo polyfills accept them.
if not hasattr(sys, "get_int_max_str_digits"):

    def get_int_max_str_digits() -> int:
        return 4300

    def set_int_max_str_digits(maxdigits: int) -> None:
        return None

    sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
    sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]

__version__ = "0.1.0"
