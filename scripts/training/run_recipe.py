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
"""
Training script for Megatron-Bridge recipes.
This script runs inside the container and handles the actual training execution.
"""

import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PERFORMANCE_SCRIPT_DIR = SCRIPT_DIR.parent / "performance"
for path in (str(SCRIPT_DIR), str(PERFORMANCE_SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from argument_parser import parse_cli_args  # noqa: E402
from recipe_runner import (  # noqa: E402
    apply_cli_overrides,
    apply_determinism,
    infer_train_mode,
    load_forward_step,
    load_library_recipe_by_family,
    load_perf_recipe_by_name,
    load_recipe,
    run_config,
    sync_model_dataset_sequence_length,
)
from utils.overrides import set_cli_overrides, set_post_overrides, set_user_overrides  # noqa: E402

from megatron.bridge.recipes.utils.dataset_utils import apply_dataset_override, infer_mode_from_dataset  # noqa: E402


PERF_RECIPE_PRECISION_PATTERN = re.compile(r"_\d+gpu_[^_]+_(bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_|_config)")


def _default_step_name(args) -> str:
    """Select the default forward step from the requested training domain."""
    if args.step_func is not None:
        return args.step_func
    if args.domain == "vlm":
        return "vlm_step"
    if args.domain == "qwen3vl":
        return "qwen3_vl_step"
    if args.domain == "diffusion":
        recipe_name = args.recipe or args.model_recipe_name or ""
        return "flux_step" if args.model_family_name == "flux" or recipe_name.startswith("flux") else "wan_step"
    return "gpt_step"


def _validate_selector_args(args) -> None:
    """Validate legacy recipe selector arguments."""
    required_fields = ("model_family_name", "model_recipe_name", "num_gpus", "gpu")
    missing = [field for field in required_fields if getattr(args, field) is None]
    if missing:
        formatted = ", ".join(f"--{field}" for field in missing)
        raise ValueError(f"Missing required recipe selector arguments: {formatted}. Pass them or use --recipe.")


def _apply_performance_compatibility_overrides(recipe, args):
    """Apply compatibility behavior shared by full-name and selector perf recipes."""
    precision = args.compute_dtype
    if args.recipe is not None:
        match = PERF_RECIPE_PRECISION_PATTERN.search(args.recipe)
        if match is not None:
            precision = match.group(1)

    if precision == "bf16" and recipe.optimizer.optimizer == "adam":
        recipe.optimizer.use_precision_aware_optimizer = True
    return recipe


def _run_full_library_recipe(args, cli_overrides: list[str]) -> None:
    """Run a library recipe selected by its full function name."""
    recipe = load_recipe(
        args.recipe,
        args.peft_scheme,
        args.packed_sequence,
        args.seq_length,
        args.hf_path,
        source="recipes",
    )

    if args.dataset is not None:
        mode = infer_mode_from_dataset(args.dataset)
        recipe = apply_dataset_override(
            recipe,
            dataset_type=args.dataset,
            packed_sequence=args.packed_sequence,
            seq_length=args.seq_length,
            cli_overrides=cli_overrides,
        )
    else:
        mode = infer_train_mode(args.recipe)

    recipe = apply_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args, recipe_source="library")
    recipe = apply_determinism(recipe, deterministic=args.deterministic)
    recipe = sync_model_dataset_sequence_length(recipe)

    forward_step = load_forward_step(_default_step_name(args), mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        dryrun_num_gpus=args.num_gpus,
    )


def _run_library_selector(args, cli_overrides: list[str]) -> None:
    """Run a library recipe through the legacy family/name selector."""
    _validate_selector_args(args)
    recipe = load_library_recipe_by_family(
        model_family_name=args.model_family_name,
        model_recipe_name=args.model_recipe_name,
        train_task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=args.config_variant,
        wandb_experiment_name=args.wandb_experiment_name,
        peft_scheme=args.peft_scheme,
    )
    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args, recipe_source="library")
    recipe = apply_determinism(recipe, deterministic=args.deterministic)

    mode = "pretrain" if args.task == "pretrain" else "finetune"
    forward_step = load_forward_step(_default_step_name(args), mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        dryrun_num_gpus=args.num_gpus,
    )


def _run_benchmark(args, cli_overrides: list[str]) -> None:
    """Run a flat performance recipe selected by full name or legacy dimensions."""
    if args.use_recipes:
        raise ValueError("--use_recipes is not valid with scripts/performance/run_script.py.")

    if args.recipe is not None:
        recipe = load_recipe(args.recipe, source="perf_recipes")
        mode = infer_train_mode(args.recipe)
    else:
        _validate_selector_args(args)
        recipe = load_perf_recipe_by_name(
            model_recipe_name=args.model_recipe_name,
            task=args.task,
            num_gpus=args.num_gpus,
            gpu=args.gpu,
            precision=args.compute_dtype,
            config_variant=args.config_variant,
        )
        mode = "pretrain" if args.task == "pretrain" else "finetune"

    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args, recipe_source="performance")
    recipe = _apply_performance_compatibility_overrides(recipe, args)
    if args.recipe is None:
        recipe = set_post_overrides(
            recipe,
            args.model_family_name,
            args.model_recipe_name,
            args.gpu,
            args.num_gpus,
            args.compute_dtype,
            args.task,
            user_gbs=args.global_batch_size,
            config_variant=args.config_variant,
        )
    recipe = apply_determinism(recipe, deterministic=args.deterministic)

    if getattr(recipe.ddp, "nccl_ub", False):
        os.environ["NCCL_NVLS_ENABLE"] = "1"
        os.environ["NCCL_CTA_POLICY"] = "1"

    forward_step = load_forward_step(_default_step_name(args), mode=mode)
    run_config(
        config=recipe,
        mode=mode,
        step_func=forward_step,
        dryrun=args.dryrun,
        save_config_filepath=args.save_config_filepath,
        barrier_before_destroy=True,
        dryrun_num_gpus=args.num_gpus,
        dump_environment=args.dump_env,
    )


def main(*, benchmark: bool = False) -> None:
    """Main entry point for the training script."""

    # Parse known args and capture unknown ones for Hydra-style config overrides
    # (e.g. model.hidden_size=15360 model.num_moe_experts=8)
    parser = parse_cli_args()
    args, cli_overrides = parser.parse_known_args()

    if benchmark:
        _run_benchmark(args, cli_overrides)
        return

    if args.recipe is not None:
        _run_full_library_recipe(args, cli_overrides)
        return

    _run_library_selector(args, cli_overrides)


if __name__ == "__main__":
    main()
