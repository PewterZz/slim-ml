from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MoESpec:
    num_experts: int
    top_k: int
    shared_experts: int
    expert_intermediate: int


@dataclass(frozen=True)
class AttentionSpec:
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float
    active_params_b: float
    num_layers: int
    hidden_size: int
    attention: AttentionSpec
    moe: Optional[MoESpec] = None
    notes: str = ""

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    def expert_bytes(self, quant_bits: int = 4) -> int:
        if not self.is_moe:
            return 0
        m = self.moe
        params = 3 * self.hidden_size * m.expert_intermediate
        return params * quant_bits // 8

    def active_bytes_per_token(self, quant_bits: int = 4) -> int:
        if not self.is_moe:
            return int(self.active_params_b * 1e9 * quant_bits / 8)
        m = self.moe
        per_layer = (m.top_k + m.shared_experts) * self.expert_bytes(quant_bits)
        return per_layer * self.num_layers


ARCH_REGISTRY: dict[str, ModelSpec] = {
    "qwen3-next-80b-a3b": ModelSpec(
        name="qwen3-next-80b-a3b",
        total_params_b=80.0,
        active_params_b=3.0,
        num_layers=48,
        hidden_size=2048,
        attention=AttentionSpec(num_heads=16, num_kv_heads=2, head_dim=256),
        moe=MoESpec(num_experts=512, top_k=10, shared_experts=1, expert_intermediate=512),
        notes="Hybrid: 3x(GatedDeltaNet->MoE) + 1x(GatedAttention->MoE), repeated 12x",
    ),
    "qwen3.6-35b-a3b": ModelSpec(
        name="qwen3.6-35b-a3b",
        total_params_b=35.0,
        active_params_b=3.0,
        num_layers=40,
        hidden_size=2048,
        attention=AttentionSpec(num_heads=16, num_kv_heads=2, head_dim=256),
        moe=MoESpec(num_experts=256, top_k=8, shared_experts=1, expert_intermediate=512),
        notes="Hybrid: 3x(GatedDeltaNet->MoE) + 1x(GatedAttention->MoE), repeated 10x",
    ),
}
