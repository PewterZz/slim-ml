"""Routing-skew probe — superseded by tools/routing_observe.py.

Kept as a pointer so anyone who remembered the old path lands on the new one.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("routing_probe: superseded by tools/routing_observe.py.")
    print("Run: python tools/routing_observe.py --model <mlx MoE model>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
