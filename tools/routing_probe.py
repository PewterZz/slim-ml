"""Routing-skew probe.

Measures how concentrated a MoE model's expert routing is on a sample workload.
If routing is uniform, expert caching buys less. If routing is Zipfian-skewed
(expected), a 15-20% hot set can capture 40-60% of routes — that's the whole
thesis behind the expert_cache technique.

STATUS: stub. Needs backend-side routing hook (see Backend.set_route_callback).
When MLXBackend.supports_routing_hooks() returns True this script becomes real.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("routing_probe: not implemented yet.")
    print("Blocker: MLXBackend.set_route_callback() is NotImplemented.")
    print("Next step: extend MLXBackend to expose router outputs per layer, then run this")
    print("           over a ~1000-token sample and plot routing CDF.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
