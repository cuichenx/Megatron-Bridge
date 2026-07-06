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

"""Unit tests for Qwen3-Omni training entry wiring in recipe_runner.py."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_recipe_runner_module():
    """Load recipe_runner.py with lightweight stub modules for local unit testing."""

    script_path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "recipe_runner.py"
    module_name = "test_recipe_runner_qwen3_omni_module"

    megatron_module = _package("megatron")
    bridge_module = _package("megatron.bridge")
    models_module = _package("megatron.bridge.models")
    qwen_omni_models_module = _package("megatron.bridge.models.qwen_omni")
    qwen_vl_models_module = _package("megatron.bridge.models.qwen_vl")
    stepfun_models_module = _package("megatron.bridge.models.stepfun")
    diffusion_module = _package("megatron.bridge.diffusion")
    diffusion_models_module = _package("megatron.bridge.diffusion.models")
    flux_models_module = _package("megatron.bridge.diffusion.models.flux")
    wan_models_module = _package("megatron.bridge.diffusion.models.wan")
    recipes_module = _package("megatron.bridge.recipes")
    recipes_utils_module = _package("megatron.bridge.recipes.utils")
    training_module = _package("megatron.bridge.training")
    training_utils_module = _package("megatron.bridge.training.utils")
    utils_module = _package("megatron.bridge.utils")

    qwen3_omni_step = types.ModuleType("megatron.bridge.models.qwen_omni.qwen3_omni_step")
    qwen3_omni_step.forward_step = Mock(name="qwen3_omni_forward_step")

    qwen3_vl_step = types.ModuleType("megatron.bridge.models.qwen_vl.qwen3_vl_step")
    qwen3_vl_step.forward_step = object()

    step37_flickr8k_step = types.ModuleType("megatron.bridge.models.stepfun.step37_flickr8k_step")
    step37_flickr8k_step.forward_step = object()

    gpt_step = types.ModuleType("megatron.bridge.training.gpt_step")
    gpt_step.forward_step = object()

    vlm_step = types.ModuleType("megatron.bridge.training.vlm_step")
    vlm_step.forward_step = object()

    llava_step = types.ModuleType("megatron.bridge.training.llava_step")
    llava_step.forward_step = object()

    nemotron_omni_step = types.ModuleType("megatron.bridge.training.nemotron_omni_step")
    nemotron_omni_step.forward_step = object()

    audio_lm_step = types.ModuleType("megatron.bridge.training.audio_lm_step")
    audio_lm_step.forward_step = object()

    flux_step = types.ModuleType("megatron.bridge.diffusion.models.flux.flux_step")

    class FluxForwardStep:
        pass

    flux_step.FluxForwardStep = FluxForwardStep

    wan_step = types.ModuleType("megatron.bridge.diffusion.models.wan.wan_step")

    class WanForwardStep:
        def __init__(self, mode=None):
            self.mode = mode

    wan_step.WanForwardStep = WanForwardStep

    determinism_utils_module = types.ModuleType("megatron.bridge.recipes.utils.determinism_utils")
    determinism_utils_module.apply_determinism_overrides = Mock(name="apply_determinism_overrides")

    naming_module = types.ModuleType("megatron.bridge.recipes.utils.naming")
    naming_module.recipe_variant_suffix = lambda config_variant: (
        "" if config_variant is None or config_variant.lower() == "v2" else f"_{config_variant.lower()}"
    )
    naming_module.recipe_function_name = (
        lambda *, model_recipe_name, task, num_gpus, gpu, precision, config_variant=None: (
            f"{model_recipe_name}_{task}_{num_gpus}gpu_{gpu}_{precision}"
            f"{naming_module.recipe_variant_suffix(config_variant)}_config"
        )
    )

    finetune_module = types.ModuleType("megatron.bridge.training.finetune")
    finetune_module.finetune = Mock(name="finetune")

    pretrain_module = types.ModuleType("megatron.bridge.training.pretrain")
    pretrain_module.pretrain = Mock(name="pretrain")

    class TokenizerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_module = types.ModuleType("megatron.bridge.training.config")
    config_module.ConfigContainer = object
    config_module.TokenizerConfig = TokenizerConfig
    config_module.apply_environment_variables = Mock(name="apply_environment_variables")
    config_module.runtime_config_update = Mock(name="runtime_config_update")

    omegaconf_module = types.ModuleType("megatron.bridge.training.utils.omegaconf_utils")
    omegaconf_module.process_config_with_overrides = lambda config, cli_overrides=None: config

    common_utils_module = types.ModuleType("megatron.bridge.utils.common_utils")
    common_utils_module.get_rank_safe = lambda: 1

    torch_module = types.ModuleType("torch")
    torch_module.distributed = types.SimpleNamespace(
        barrier=Mock(name="barrier"),
        destroy_process_group=Mock(name="destroy_process_group"),
        is_initialized=lambda: False,
    )

    stub_modules = {
        "torch": torch_module,
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.bridge.models": models_module,
        "megatron.bridge.models.qwen_omni": qwen_omni_models_module,
        "megatron.bridge.models.qwen_vl": qwen_vl_models_module,
        "megatron.bridge.models.stepfun": stepfun_models_module,
        "megatron.bridge.diffusion": diffusion_module,
        "megatron.bridge.diffusion.models": diffusion_models_module,
        "megatron.bridge.diffusion.models.flux": flux_models_module,
        "megatron.bridge.diffusion.models.wan": wan_models_module,
        "megatron.bridge.recipes": recipes_module,
        "megatron.bridge.recipes.utils": recipes_utils_module,
        "megatron.bridge.recipes.utils.determinism_utils": determinism_utils_module,
        "megatron.bridge.recipes.utils.naming": naming_module,
        "megatron.bridge.training": training_module,
        "megatron.bridge.training.utils": training_utils_module,
        "megatron.bridge.utils": utils_module,
        "megatron.bridge.utils.common_utils": common_utils_module,
        "megatron.bridge.diffusion.models.flux.flux_step": flux_step,
        "megatron.bridge.diffusion.models.wan.wan_step": wan_step,
        "megatron.bridge.models.qwen_omni.qwen3_omni_step": qwen3_omni_step,
        "megatron.bridge.models.qwen_vl.qwen3_vl_step": qwen3_vl_step,
        "megatron.bridge.models.stepfun.step37_flickr8k_step": step37_flickr8k_step,
        "megatron.bridge.training.audio_lm_step": audio_lm_step,
        "megatron.bridge.training.gpt_step": gpt_step,
        "megatron.bridge.training.vlm_step": vlm_step,
        "megatron.bridge.training.llava_step": llava_step,
        "megatron.bridge.training.nemotron_omni_step": nemotron_omni_step,
        "megatron.bridge.training.finetune": finetune_module,
        "megatron.bridge.training.pretrain": pretrain_module,
        "megatron.bridge.training.config": config_module,
        "megatron.bridge.training.utils.omegaconf_utils": omegaconf_module,
    }

    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    test_handles = {
        "finetune": finetune_module.finetune,
        "apply_environment_variables": config_module.apply_environment_variables,
        "omni_forward_step": qwen3_omni_step.forward_step,
        "pretrain": pretrain_module.pretrain,
        "wan_forward_step": WanForwardStep,
    }
    return module, test_handles


def _load_run_recipe_module():
    """Load run_recipe.py with its external modules stubbed."""
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "training" / "run_recipe.py"
    module_name = "test_unified_run_recipe_module"

    argument_parser = types.ModuleType("argument_parser")
    argument_parser.parse_cli_args = Mock(name="parse_cli_args")

    recipe_runner = types.ModuleType("recipe_runner")
    for name in (
        "apply_cli_overrides",
        "apply_determinism",
        "apply_tokenizer_override",
        "dump_env_rank0",
        "infer_train_mode",
        "load_forward_step",
        "load_library_recipe_by_family",
        "load_perf_recipe_by_name",
        "load_recipe",
        "run_config",
        "sync_model_dataset_sequence_length",
    ):
        setattr(recipe_runner, name, Mock(name=name))

    utils_module = _package("utils")
    overrides_module = types.ModuleType("utils.overrides")
    overrides_module.set_cli_overrides = Mock(name="set_cli_overrides")
    overrides_module.set_post_overrides = Mock(name="set_post_overrides")
    overrides_module.set_user_overrides = Mock(name="set_user_overrides")

    megatron_module = _package("megatron")
    bridge_module = _package("megatron.bridge")
    recipes_module = _package("megatron.bridge.recipes")
    recipes_utils_module = _package("megatron.bridge.recipes.utils")
    dataset_utils_module = types.ModuleType("megatron.bridge.recipes.utils.dataset_utils")
    dataset_utils_module.apply_dataset_override = Mock(name="apply_dataset_override")
    dataset_utils_module.infer_mode_from_dataset = Mock(name="infer_mode_from_dataset")

    stub_modules = {
        "argument_parser": argument_parser,
        "recipe_runner": recipe_runner,
        "utils": utils_module,
        "utils.overrides": overrides_module,
        "megatron": megatron_module,
        "megatron.bridge": bridge_module,
        "megatron.bridge.recipes": recipes_module,
        "megatron.bridge.recipes.utils": recipes_utils_module,
        "megatron.bridge.recipes.utils.dataset_utils": dataset_utils_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    return module


class TestRecipeRunnerQwen3Omni:
    """Tests for wiring Qwen3-Omni into the shared recipe runner."""

    def test_load_forward_step_returns_qwen3_omni_handler(self):
        """The shared registry should expose qwen3_omni_step."""

        module, handles = _load_recipe_runner_module()

        module.STEP_FUNCTIONS["qwen3_omni_step"] = handles["omni_forward_step"]
        assert module.load_forward_step("qwen3_omni_step") is handles["omni_forward_step"]

    def test_class_based_forward_step_receives_mode(self):
        """Class-based diffusion steps should still receive the selected train mode."""

        module, handles = _load_recipe_runner_module()
        module.STEP_FUNCTIONS["wan_step"] = handles["wan_forward_step"]

        forward_step = module.load_forward_step("wan_step", mode="finetune")

        assert forward_step.mode == "finetune"

    def test_run_config_routes_qwen3_omni_step_to_finetune(self):
        """The shared runner should pass the Omni step function into finetune."""

        module, handles = _load_recipe_runner_module()
        config = object()

        module.run_config(config=config, mode="finetune", step_func=handles["omni_forward_step"])

        handles["finetune"].assert_called_once_with(config=config, forward_step_func=handles["omni_forward_step"])
        handles["pretrain"].assert_not_called()

    def test_run_config_applies_environment_before_dump_and_training(self):
        """Recipe environment defaults should be visible to diagnostics and training."""
        module, handles = _load_recipe_runner_module()
        events = []
        handles["apply_environment_variables"].side_effect = lambda config: events.append("environment")
        module.dump_env_rank0 = Mock(side_effect=lambda: events.append("dump"))
        handles["finetune"].side_effect = lambda **kwargs: events.append("training")

        module.run_config(config=object(), mode="finetune", step_func=object(), dump_environment=True)

        assert events == ["environment", "dump", "training"]

    def test_dry_run_uses_target_gpu_count_over_slurm_allocation(self, monkeypatch, tmp_path):
        """Dry-run topology validation should use the requested target GPU count."""
        module, _ = _load_recipe_runner_module()
        config = SimpleNamespace(to_yaml=Mock(), print_yaml=Mock())
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "7")
        monkeypatch.setenv("SLURM_NTASKS", "1")
        monkeypatch.setenv("SLURM_PROCID", "7")

        module.run_config(
            config=config,
            mode="pretrain",
            step_func=object(),
            dryrun=True,
            save_config_filepath=str(tmp_path / "config.yaml"),
            dryrun_num_gpus=8,
        )

        assert module.os.environ["WORLD_SIZE"] == "8"
        assert module.os.environ["RANK"] == "0"
        module.runtime_config_update.assert_called_once_with(config)

    def test_load_recipe_auto_prefers_library_recipe(self):
        """Auto source resolution should check library recipes before flat perf recipes."""

        module, _ = _load_recipe_runner_module()
        library_config = object()
        perf_config = object()
        module.find_library_recipe = Mock(return_value=lambda: library_config)
        module.find_perf_recipe = Mock(return_value=lambda: perf_config)

        cfg = module.load_recipe("shared_name_config", source="auto")

        assert cfg is library_config
        module.find_perf_recipe.assert_not_called()


class TestUnifiedRunRecipeRouting:
    """Tests for normal and benchmark mode dispatch."""

    def test_benchmark_entrypoint_routes_to_performance_path(self):
        module = _load_run_recipe_module()
        args = SimpleNamespace(recipe=None)
        parser = Mock()
        parser.parse_known_args.return_value = (args, ["model.hidden_size=128"])
        module.parse_cli_args = Mock(return_value=parser)
        module._run_benchmark = Mock()
        module._run_library_selector = Mock()

        module.main(benchmark=True)

        module._run_benchmark.assert_called_once_with(args, ["model.hidden_size=128"])
        module._run_library_selector.assert_not_called()

    def test_selector_without_benchmark_routes_to_library_path(self):
        module = _load_run_recipe_module()
        args = SimpleNamespace(recipe=None)
        parser = Mock()
        parser.parse_known_args.return_value = (args, [])
        module.parse_cli_args = Mock(return_value=parser)
        module._run_benchmark = Mock()
        module._run_library_selector = Mock()

        module.main()

        module._run_library_selector.assert_called_once_with(args, [])
        module._run_benchmark.assert_not_called()

    def test_full_recipe_without_benchmark_routes_to_library_path(self):
        module = _load_run_recipe_module()
        args = SimpleNamespace(recipe="llama3_8b_sft_config")
        parser = Mock()
        parser.parse_known_args.return_value = (args, [])
        module.parse_cli_args = Mock(return_value=parser)
        module._run_benchmark = Mock()
        module._run_full_library_recipe = Mock()

        module.main()

        module._run_full_library_recipe.assert_called_once_with(args, [])
        module._run_benchmark.assert_not_called()

    def test_benchmark_preserves_recipe_dataset_when_data_is_omitted(self):
        module = _load_run_recipe_module()
        recipe = SimpleNamespace(
            ddp=SimpleNamespace(nccl_ub=False),
            optimizer=SimpleNamespace(optimizer="adam", use_precision_aware_optimizer=False),
        )
        args = SimpleNamespace(
            use_recipes=False,
            recipe="llama3_8b_pretrain_8gpu_h100_bf16_config",
            data=None,
            dump_env=False,
            deterministic=False,
            compute_dtype="fp8_cs",
            step_func=None,
            domain="llm",
            dryrun=False,
            save_config_filepath=None,
            num_gpus=8,
        )
        module.load_recipe = Mock(return_value=recipe)
        module.infer_train_mode = Mock(return_value="pretrain")
        module.set_cli_overrides = Mock(side_effect=lambda cfg, _: cfg)
        module.set_user_overrides = Mock(side_effect=lambda cfg, _, **__: cfg)
        module.apply_determinism = Mock(side_effect=lambda cfg, **_: cfg)
        module.load_forward_step = Mock(return_value=object())
        module.run_config = Mock()

        module._run_benchmark(args, [])

        assert args.data is None
        assert recipe.optimizer.use_precision_aware_optimizer is True
        module.set_user_overrides.assert_called_once_with(recipe, args, recipe_source="performance")
