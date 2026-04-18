"""Direct unit test of the route-observation monkey-patch.

Constructs a bare SwitchGLU (no model download) so the hook can be exercised
fast on any machine with mlx-lm installed.
"""
from __future__ import annotations

import io
import contextlib

import pytest


mx = pytest.importorskip("mlx.core")
switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")

from slim_ml.backend import _ROUTE_STATE, _patch_switch_class_once


def test_hook_fires_and_cleans_up():
    inst = switch_layers.SwitchGLU(8, 16, 4)
    hits: list[tuple[int, tuple[int, ...]]] = []

    def cb(layer_idx, ids, weights):
        hits.append((layer_idx, tuple(ids)))

    _patch_switch_class_once(type(inst))
    _ROUTE_STATE[id(inst)] = (7, cb)

    x = mx.random.normal(shape=(1, 8))
    idx = mx.array([[0, 2]], dtype=mx.uint32)
    out = inst(x, idx)
    mx.eval(out)

    assert hits, "hook did not fire"
    assert hits[0] == (7, (0, 2))

    _ROUTE_STATE.pop(id(inst))
    _ = inst(x, idx)
    mx.eval(_)
    assert len(hits) == 1, "hook fired after cleanup"


def test_callback_errors_surface_to_stderr():
    inst = switch_layers.SwitchGLU(8, 16, 4)

    def bad_cb(layer_idx, ids, weights):
        raise ValueError("intentional test error")

    _patch_switch_class_once(type(inst))
    _ROUTE_STATE[id(inst)] = (3, bad_cb)

    x = mx.random.normal(shape=(1, 8))
    idx = mx.array([[1, 3]], dtype=mx.uint32)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            out = inst(x, idx)
            mx.eval(out)
    finally:
        _ROUTE_STATE.pop(id(inst), None)

    assert "intentional test error" in err.getvalue()
