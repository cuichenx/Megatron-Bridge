# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for scripts/performance/perf_plugins.py PerfEnvPlugin determinism wiring."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    import nemo_run  # noqa: F401

    HAS_NEMO_RUN = True
except ImportError:
    HAS_NEMO_RUN = False

pytestmark = pytest.mark.skipif(not HAS_NEMO_RUN, reason="nemo_run not installed")

if HAS_NEMO_RUN:
    import perf_plugins
    from perf_plugins import PerfEnvPlugin


def test_set_determinism_env_vars_writes_three_keys():
    plugin = PerfEnvPlugin(
        deterministic=True,
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        gpu="h100",
        compute_dtype="bf16",
        train_task="pretrain",
        num_gpus=64,
    )
    executor = MagicMock()
    executor.env_vars = {}

    plugin._set_determinism_env_vars(executor)

    assert executor.env_vars["NCCL_ALGO"] == "Ring"
    assert executor.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert executor.env_vars["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_perf_env_plugin_num_gpus_is_optional():
    plugin = PerfEnvPlugin(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        gpu="h100",
        compute_dtype="bf16",
        train_task="pretrain",
    )

    assert plugin.num_gpus is None


def test_recipe_environment_defaults_preserve_explicit_executor_values():
    """Recipe defaults and plugin fallbacks must not replace explicit launcher values."""
    plugin = PerfEnvPlugin(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype="bf16",
        train_task="pretrain",
        num_gpus=256,
    )
    executor = MagicMock()
    executor.env_vars = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "48",
        "USE_MNNVL": "custom",
    }

    plugin._set_layernorm_sm_margin(MagicMock(), executor, True, 16)
    plugin._set_nvl_domain_size(MagicMock(), executor, "hybridep", "gb200", 64)

    assert executor.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == "48"
    assert executor.env_vars["USE_MNNVL"] == "custom"
    assert executor.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "64"


def test_plugin_added_environment_is_forced_into_slurm_container(monkeypatch):
    """Slurm container_env should reflect the final plugin environment keys."""

    class FakeSlurmExecutor:
        def __init__(self):
            self.env_vars = {"TORCHINDUCTOR_WORKER_START": "fork"}
            self.container_env = []

    monkeypatch.setattr(perf_plugins, "SlurmExecutor", FakeSlurmExecutor)
    executor = FakeSlurmExecutor()

    PerfEnvPlugin._sync_slurm_container_env(executor)

    assert executor.container_env == ["TORCHINDUCTOR_WORKER_START"]
