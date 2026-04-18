"""Profile MoE routing on a small MoE via slim-ml's route-observation hook.

Stage 0 of expert caching: confirm experimentally that expert routing is
skewed enough (Zipfian-ish) that a hot-set covering some small fraction of
experts captures a much larger fraction of routing events. This is the
premise of expert caching; without this the rest is moot.

Usage:
    python tools/routing_observe.py \
        --model mlx-community/OLMoE-1B-7B-0125-Instruct-4bit \
        --max-tokens 128

Default prompt is a multi-topic essay so routing touches a broad surface.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

from slim_ml.backend import GenerationSettings, MLXBackend
from slim_ml.budget import StaticBudget, auto_detect_limits
from slim_ml.runtime import Session


def capture_at(sorted_hits: list[int], total: int, frac_experts: float) -> float:
    if total == 0 or not sorted_hits:
        return 0.0
    k = max(1, int(len(sorted_hits) * frac_experts))
    return sum(sorted_hits[:k]) / total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--prompt",
        default=(
            "Write a short essay covering three topics in one paragraph "
            "each: the history of Python, the physics of neutron stars, "
            "and the culinary tradition of Sichuan peppercorns. Use "
            "specific names, dates, and technical terms where relevant."
        ),
    )
    args = p.parse_args()

    be = MLXBackend()
    budget = StaticBudget(auto_detect_limits())
    print(f"[load] {args.model}")
    be.load(args.model, None, budget)

    if not be.supports_routing_hooks():
        print("[error] model has no MoE layers with switch_mlp — nothing to observe")
        return 2

    per_layer: dict[int, Counter] = defaultdict(Counter)
    total_events = {"n": 0}

    def on_route(layer_idx: int, expert_ids, weights):
        per_layer[layer_idx].update(expert_ids)
        total_events["n"] += len(expert_ids)

    installed = be.set_route_callback(on_route)
    print(f"[hook] installed on {installed} MoE layer(s)")

    session = Session(backend=be, budget=budget)
    settings = GenerationSettings(max_tokens=args.max_tokens, temperature=args.temperature)

    try:
        text_tokens = 0
        for tok in session.generate(args.prompt, settings):
            text_tokens += 1
        print(f"[gen] produced {text_tokens} tokens")
    finally:
        be.set_route_callback(None)

    print(f"\n[routing] total events={total_events['n']} across {len(per_layer)} MoE layer(s)")

    global_hits: Counter = Counter()
    for lidx, c in per_layer.items():
        for eid, n in c.items():
            global_hits[(lidx, eid)] += n

    sorted_global = sorted(global_hits.values(), reverse=True)
    total_global = sum(sorted_global)
    n_slots = len(global_hits)
    print(f"[global] {n_slots} (layer,expert) slots observed")
    for frac in (0.05, 0.10, 0.20, 0.50):
        cap = capture_at(sorted_global, total_global, frac)
        print(f"  top {frac*100:4.0f}% experts capture {cap*100:5.1f}% of routes")

    print("\n[per-layer top-10% capture]")
    print("layer | experts_seen | top10%_capture")
    for lidx in sorted(per_layer):
        vals = sorted(per_layer[lidx].values(), reverse=True)
        seen = len(vals)
        total = sum(vals)
        cap10 = capture_at(vals, total, 0.10)
        print(f" {lidx:4d} |  {seen:6d}      |    {cap10*100:5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
