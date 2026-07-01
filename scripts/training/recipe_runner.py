#!/usr/bin/env python3
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

"""Shared helpers for recipe-based training entry points."""

import functools
import importlib
import inspect
import logging
import os
import pkgutil
from collections.abc import Callable
from typing import Literal, cast

import torch

import megatron.bridge.recipes as recipes
from megatron.bridge.recipes.utils.determinism_utils import apply_determinism_overrides
from megatron.bridge.recipes.utils.naming import (
    recipe_function_name,
    recipe_variant_suffix,
)
from megatron.bridge.training.config import ConfigContainer, TokenizerConfig, runtime_config_update
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.utils.common_utils import get_rank_safe


logger = logging.getLogger(__name__)

RecipeSource = Literal["auto", "recipes", "perf_recipes"]

StepFunctionEntry = Callable | tuple[str, str]

STEP_FUNCTIONS: dict[str, StepFunctionEntry] = {
    "audio_lm_step": ("megatron.bridge.training.audio_lm_step", "forward_step"),
    "gpt_step": ("megatron.bridge.training.gpt_step", "forward_step"),
    "vlm_step": ("megatron.bridge.training.vlm_step", "forward_step"),
    "qwen3_omni_step": ("megatron.bridge.models.qwen_omni.qwen3_omni_step", "forward_step"),
    "qwen3_vl_step": ("megatron.bridge.models.qwen_vl.qwen3_vl_step", "forward_step"),
    "step37_flickr8k_step": ("megatron.bridge.models.stepfun.step37_flickr8k_step", "forward_step"),
    "llava_step": ("megatron.bridge.training.llava_step", "forward_step"),
    "nemotron_omni_step": ("megatron.bridge.training.nemotron_omni_step", "forward_step"),
    "flux_step": ("megatron.bridge.diffusion.models.flux.flux_step", "FluxForwardStep"),
    "wan_step": ("megatron.bridge.diffusion.models.wan.wan_step", "WanForwardStep"),
}

TRAIN_FUNCTIONS = {
    "pretrain": pretrain,
    "finetune": finetune,
}

ERR_UNKNOWN_STEP = "Unknown step type: {step_type}. Choose from: {choices}"
ERR_INFER_MODE_FAILED = (
    "Unable to infer training mode. "
    "Pass --dataset to specify the dataset type, or include 'pretrain' or 'finetune' "
    "(or 'sft'/'peft'/'lora') in the recipe name."
)


def _recipe_kwargs_for_signature(
    config_builder: Callable,
    *,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
) -> dict[str, object]:
    """Build kwargs accepted by a recipe function."""
    try:
        sig = inspect.signature(config_builder)
        params = sig.parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        accepts_peft = "peft" in params or has_var_keyword
        accepts_peft_scheme = "peft_scheme" in params
        accepts_packed_sequence = "packed_sequence" in params or has_var_keyword
        accepts_seq_length = "seq_length" in params or has_var_keyword
        accepts_hf_path = "hf_path" in params or has_var_keyword
    except (ValueError, TypeError):
        accepts_peft = True
        accepts_peft_scheme = False
        accepts_packed_sequence = False
        accepts_seq_length = False
        accepts_hf_path = False

    kwargs: dict[str, object] = {}
    if peft_scheme is not None:
        if accepts_peft_scheme:
            kwargs["peft_scheme"] = peft_scheme
        elif accepts_peft:
            kwargs["peft"] = peft_scheme
    if accepts_packed_sequence and packed_sequence:
        kwargs["packed_sequence"] = packed_sequence
    if accepts_seq_length and seq_length is not None:
        kwargs["seq_length"] = seq_length
    if accepts_hf_path and hf_path is not None:
        kwargs["hf_path"] = hf_path
    return kwargs


def _load_with_optional_kwargs(
    config_builder: Callable,
    *,
    peft_scheme: str | None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
) -> ConfigContainer:
    """Call a recipe function with only the optional kwargs it accepts."""
    kwargs = _recipe_kwargs_for_signature(
        config_builder,
        peft_scheme=peft_scheme,
        packed_sequence=packed_sequence,
        seq_length=seq_length,
        hf_path=hf_path,
    )
    try:
        return config_builder(**kwargs)
    except TypeError:
        return config_builder()


@functools.lru_cache(maxsize=1)
def perf_recipe_family_modules() -> tuple[str, ...]:
    """Return import paths for flat performance recipe family packages."""
    import megatron.bridge.perf_recipes as perf_recipes

    module_names = [
        f"{perf_recipes.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(perf_recipes.__path__)
        if module_info.ispkg and not module_info.name.startswith("_")
    ]
    return tuple(sorted(module_names))


def find_perf_recipe(recipe_name: str) -> Callable[[], ConfigContainer] | None:
    """Find a flat perf recipe function by exported function name."""
    for module_name in perf_recipe_family_modules():
        recipe_fn = getattr(importlib.import_module(module_name), recipe_name, None)
        if callable(recipe_fn):
            return cast(Callable[[], ConfigContainer], recipe_fn)
    return None


@functools.lru_cache(maxsize=1)
def library_h100_modules() -> tuple[str, ...]:
    """Return import paths for H100 library recipe alias packages."""
    module_names = []
    for module_info in pkgutil.iter_modules(recipes.__path__):
        if not module_info.ispkg or module_info.name.startswith("_") or module_info.name == "utils":
            continue
        h100_module_name = f"{recipes.__name__}.{module_info.name}.h100"
        try:
            importlib.import_module(h100_module_name)
        except ModuleNotFoundError as exc:
            if exc.name != h100_module_name:
                raise
            continue
        module_names.append(h100_module_name)
    return tuple(sorted(module_names))


def find_library_recipe(recipe_name: str, *, model_family_name: str | None = None) -> Callable | None:
    """Find a library recipe function by legacy or H100-style exported function name."""
    if hasattr(recipes, recipe_name):
        recipe_fn = getattr(recipes, recipe_name)
        if callable(recipe_fn):
            return cast(Callable, recipe_fn)

    if model_family_name is not None:
        module_names = (f"{recipes.__name__}.{model_family_name}.h100",)
    else:
        module_names = library_h100_modules()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise
            continue
        recipe_fn = getattr(module, recipe_name, None)
        if callable(recipe_fn):
            return cast(Callable, recipe_fn)
    return None


def perf_recipe_variant_suffix(config_variant: str | None) -> str:
    """Return the function-name suffix used for non-canonical perf variants."""
    return recipe_variant_suffix(config_variant)


def perf_recipe_function_name(
    *,
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
) -> str:
    """Build a flat perf recipe function name from CLI dimensions."""
    return recipe_function_name(
        model_recipe_name=model_recipe_name,
        task=task,
        num_gpus=num_gpus,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
    )


def load_perf_recipe_by_name(
    *,
    model_recipe_name: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None = None,
) -> ConfigContainer:
    """Load a flat perf recipe from ``megatron.bridge.perf_recipes``."""
    name = perf_recipe_function_name(
        model_recipe_name=model_recipe_name,
        task=task,
        num_gpus=num_gpus,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
    )
    recipe_fn = find_perf_recipe(name)
    if recipe_fn is None:
        searched_modules = ", ".join(perf_recipe_family_modules()) or "none"
        raise ValueError(f"No perf recipe {name!r} found in perf recipe packages: {searched_modules}.")
    return recipe_fn()


def load_recipe(
    recipe_name: str,
    peft_scheme: str | None = None,
    packed_sequence: bool = False,
    seq_length: int | None = None,
    hf_path: str | None = None,
    source: RecipeSource = "auto",
) -> ConfigContainer:
    """Load a recipe from the library recipe package or flat perf recipe package."""
    if source in {"auto", "recipes"}:
        config_builder = find_library_recipe(recipe_name)
        if config_builder is not None:
            return _load_with_optional_kwargs(
                config_builder,
                peft_scheme=peft_scheme,
                packed_sequence=packed_sequence,
                seq_length=seq_length,
                hf_path=hf_path,
            )

    if source in {"auto", "perf_recipes"}:
        recipe_fn = find_perf_recipe(recipe_name)
        if recipe_fn is not None:
            return recipe_fn()

    if source == "recipes":
        raise AttributeError(f"Recipe '{recipe_name}' not found in megatron.bridge.recipes.")

    raise AttributeError(
        f"Recipe '{recipe_name}' not found in requested source '{source}'. "
        "Check megatron.bridge.recipes exports or megatron.bridge.perf_recipes flat recipe names."
    )


def load_library_recipe_by_family(
    *,
    model_family_name: str,
    model_recipe_name: str,
    train_task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None,
    wandb_experiment_name: str | None,
    peft_scheme: str | None = None,
) -> ConfigContainer:
    """Load a library recipe from family/name/task CLI dimensions."""
    family_pkg = importlib.import_module(f"megatron.bridge.recipes.{model_family_name}")
    h100_recipe_name = recipe_function_name(
        model_recipe_name=model_recipe_name,
        task="lora" if train_task == "peft" else train_task,
        num_gpus=num_gpus,
        gpu=gpu,
        precision=precision,
        config_variant=config_variant,
    )

    if model_recipe_name == "deepseek_v3_32nodes" and train_task == "pretrain":
        recipe_name = "deepseek_v3_pretrain_config_32nodes"
    elif train_task in ("lora", "peft"):
        recipe_name = f"{model_recipe_name}_peft_config"
    else:
        recipe_name = f"{model_recipe_name}_{train_task}_config"

    recipe_builder = find_library_recipe(h100_recipe_name, model_family_name=model_family_name)
    if recipe_builder is None:
        recipe_builder = getattr(family_pkg, recipe_name)
    cfg = _load_with_optional_kwargs(recipe_builder, peft_scheme=peft_scheme)

    if wandb_experiment_name:
        run_output_dir = os.path.join("/nemo_run", wandb_experiment_name)
        cfg.checkpoint.save = os.path.join(run_output_dir, "checkpoints")
        cfg.checkpoint.load = os.path.join(run_output_dir, "checkpoints")
        cfg.logger.tensorboard_dir = os.path.join(run_output_dir, "tb_logs")
        cfg.logger.wandb_exp_name = wandb_experiment_name
        cfg.logger.wandb_save_dir = os.path.join(run_output_dir, "wandb")

    return cfg


def load_forward_step(step_type: str, mode: str | None = None) -> Callable:
    """Load a forward-step callable by name."""
    step_key = step_type.lower()
    if step_key not in STEP_FUNCTIONS:
        raise ValueError(ERR_UNKNOWN_STEP.format(step_type=step_type, choices=", ".join(STEP_FUNCTIONS)))
    step = _load_step_function(step_key)
    if inspect.isclass(step):
        if "mode" in inspect.signature(step.__init__).parameters:
            return step(mode=mode)
        return step()
    return step


@functools.lru_cache(maxsize=None)
def _load_step_function(step_key: str) -> Callable:
    """Import a forward-step callable from the lazy step registry."""
    entry = STEP_FUNCTIONS[step_key]
    if callable(entry):
        return entry

    module_name, attribute_name = entry
    return cast(Callable, getattr(importlib.import_module(module_name), attribute_name))


def infer_train_mode(recipe_name: str) -> str:
    """Infer train mode from a recipe name."""
    lowered = recipe_name.lower()
    has_pretrain = "pretrain" in lowered
    has_finetune = "finetune" in lowered or "sft" in lowered or "peft" in lowered or "lora" in lowered
    if has_pretrain ^ has_finetune:
        return "pretrain" if has_pretrain else "finetune"
    raise ValueError(ERR_INFER_MODE_FAILED)


def apply_cli_overrides(config: ConfigContainer, cli_overrides: list[str] | None) -> ConfigContainer:
    """Apply Hydra-style CLI overrides to a ConfigContainer."""
    return process_config_with_overrides(config, cli_overrides=cli_overrides or None)


def apply_tokenizer_override(
    config: ConfigContainer,
    *,
    tokenizer_type: str | None,
    tokenizer_model: str | None,
    vocab_size: int,
) -> ConfigContainer:
    """Apply tokenizer-related CLI overrides."""
    if tokenizer_type == "NullTokenizer":
        config.tokenizer = TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size)
    elif tokenizer_type == "HuggingFaceTokenizer":
        if not tokenizer_model:
            raise ValueError("--tokenizer-model is required when using HuggingFaceTokenizer")
        config.tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=tokenizer_model)
    elif tokenizer_type == "SentencePieceTokenizer":
        if not tokenizer_model:
            raise ValueError("--tokenizer-model is required for SentencePieceTokenizer")
        config.tokenizer = TokenizerConfig(tokenizer_type="SentencePieceTokenizer", tokenizer_model=tokenizer_model)
    return config


def apply_determinism(config: ConfigContainer, *, deterministic: bool) -> ConfigContainer:
    """Apply deterministic-training config overrides when requested."""
    if deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        apply_determinism_overrides(config)
    return config


def sync_model_dataset_sequence_length(config: ConfigContainer) -> ConfigContainer:
    """Keep model sequence length aligned with dataset sequence length when both exist."""
    if (
        hasattr(config, "model")
        and config.model is not None
        and hasattr(config, "dataset")
        and config.dataset is not None
        and hasattr(config.dataset, "seq_length")
        and config.model.seq_length != config.dataset.seq_length
    ):
        config.model.seq_length = config.dataset.seq_length
    return config


def save_config(config: ConfigContainer, save_path: str) -> None:
    """Save and log a ConfigContainer YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    config.to_yaml(save_path)
    logger.info("ConfigContainer saved to: %s", os.path.abspath(save_path))
    config.print_yaml()


def run_config(
    *,
    config: ConfigContainer,
    mode: str,
    step_func: Callable,
    dryrun: bool = False,
    save_config_filepath: str | None = None,
    barrier_before_destroy: bool = False,
    dryrun_num_gpus: int | None = None,
) -> bool:
    """Run or dry-run a ConfigContainer with the selected training function.

    Returns:
        True when the caller should return without launching training.
    """
    if dryrun:
        if dryrun_num_gpus is not None:
            if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" not in os.environ:
                os.environ["WORLD_SIZE"] = str(dryrun_num_gpus)
            if "RANK" not in os.environ and "SLURM_PROCID" not in os.environ:
                os.environ["RANK"] = "0"
            runtime_config_update(config)
        save_config(config, save_config_filepath or "ConfigContainer.yaml")
        return True

    if get_rank_safe() == 0:
        logger.info("Final configuration:")
        config.print_yaml()

    train_func = TRAIN_FUNCTIONS[mode]
    train_func(config=config, forward_step_func=step_func)

    if torch.distributed.is_initialized():
        if barrier_before_destroy:
            torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    return False
