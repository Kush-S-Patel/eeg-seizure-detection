"""Compat shim for Python 3.11.0rc1 + modern PyTorch (missing int_max_str_digits)."""

from __future__ import annotations

import sys

if not hasattr(sys, "get_int_max_str_digits"):

    def get_int_max_str_digits() -> int:
        return 4300

    def set_int_max_str_digits(maxdigits: int) -> None:
        return None

    sys.get_int_max_str_digits = get_int_max_str_digits  # type: ignore[attr-defined]
    sys.set_int_max_str_digits = set_int_max_str_digits  # type: ignore[attr-defined]
