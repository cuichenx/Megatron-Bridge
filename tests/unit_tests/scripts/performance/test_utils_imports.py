"""Import behavior tests for scripts/performance/utils/utils.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_utils_import_does_not_eagerly_import_training_stack(monkeypatch):
    """Importing perf utilities should stay lightweight on headnodes."""
    monkeypatch.delitem(sys.modules, "recipe_runner", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.bridge", raising=False)

    module_path = Path(__file__).resolve().parents[4] / "scripts" / "performance" / "utils" / "utils.py"
    spec = importlib.util.spec_from_file_location("test_perf_utils_import_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    assert "recipe_runner" not in sys.modules
    assert "megatron.bridge" not in sys.modules
