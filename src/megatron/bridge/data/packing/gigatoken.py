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

"""Optional GigaToken backend for offline GPT SFT preparation."""

import importlib
import inspect
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import ModuleType
from typing import Any


_GENERATION_BLOCK = re.compile(r"{%-?\s*generation\b")


class _HuggingFaceTokenizerProxy:
    """Retain HF chat rendering while routing encoding to one backend."""

    def __init__(
        self,
        base_tokenizer: Any,
        encoding_backend: Any,
    ) -> None:
        self._base_tokenizer = base_tokenizer
        self._encoding_backend = encoding_backend
        self._apply_chat_template = type(base_tokenizer).apply_chat_template
        # Bridge deliberately uses inspect.getattr_static() when resolving
        # row-level template controls. Keep the effective template explicit on
        # the proxy instead of relying on dynamic delegation.
        self.chat_template = getattr(base_tokenizer, "chat_template", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._encoding_backend(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._encoding_backend.encode(*args, **kwargs)

    def tokenize(self, *args: Any, **kwargs: Any) -> Any:
        return self._encoding_backend.tokenize(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        # Calling the HF implementation with this proxy as ``self`` keeps Jinja
        # rendering unchanged while its eventual ``self(...)`` encode call uses
        # the selected backend above.
        return self._apply_chat_template(self, *args, **kwargs)


def _load_gigatoken() -> ModuleType:
    try:
        return importlib.import_module("gigatoken")
    except ModuleNotFoundError as error:
        if error.name != "gigatoken":
            raise
        raise ModuleNotFoundError(
            "GigaToken packed-SFT preparation requires the optional `gigatoken` package. "
            "Install it in the preparation environment before setting use_gigatoken=True."
        ) from error


def _resolve_hf_tokenizer(tokenizer: Any) -> tuple[Any, Any]:
    if getattr(tokenizer, "library", None) != "huggingface":
        raise ValueError("GigaToken packed-SFT preparation supports only HuggingFaceTokenizer.")

    tokenizer_wrapper = getattr(tokenizer, "_tokenizer", None)
    hf_tokenizer = getattr(tokenizer_wrapper, "tokenizer", None)
    if tokenizer_wrapper is None or hf_tokenizer is None:
        raise ValueError("Unable to locate the Hugging Face tokenizer used for packed-SFT preparation.")
    if not getattr(hf_tokenizer, "is_fast", False):
        raise ValueError("GigaToken packed-SFT preparation requires a fast Hugging Face tokenizer.")
    return tokenizer_wrapper, hf_tokenizer


def _supports_standard_chat_template(hf_tokenizer: Any) -> bool:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    standard_method = PreTrainedTokenizerBase.apply_chat_template
    class_method = inspect.getattr_static(type(hf_tokenizer), "apply_chat_template", None)
    instance_method = inspect.getattr_static(hf_tokenizer, "apply_chat_template", None)
    return class_method is standard_method and instance_method is standard_method


def _validate_chat_compatibility(hf_tokenizer: Any) -> None:
    if not _supports_standard_chat_template(hf_tokenizer):
        raise ValueError(
            "GigaToken packed-SFT preparation does not support tokenizers with a custom apply_chat_template method."
        )
    chat_template = getattr(hf_tokenizer, "chat_template", None)
    templates = chat_template.values() if isinstance(chat_template, Mapping) else (chat_template,)
    if any(isinstance(template, str) and _GENERATION_BLOCK.search(template) for template in templates):
        raise ValueError(
            "GigaToken packed-SFT preparation does not support chat templates with a Jinja generation block "
            "because GigaToken 0.10 does not return the offset mapping required for assistant masks."
        )


@contextmanager
def packed_sft_tokenizer_backend(
    tokenizer: Any,
    *,
    use_gigatoken: bool,
    requires_chat_template: bool,
) -> Iterator[None]:
    """Temporarily select an optional GigaToken encoding backend.

    The original tokenizer object is restored even when preparation raises.
    Disabled calls are a no-op so the established path retains its exact object
    graph and behavior.

    Args:
        tokenizer: MCore tokenizer used by GPT SFT preprocessing.
        use_gigatoken: Route supported Hugging Face encoding calls through GigaToken.
        requires_chat_template: Whether preprocessing invokes the HF chat template.

    Yields:
        None after installing the requested backend for the context lifetime.
    """
    if not use_gigatoken:
        yield None
        return

    tokenizer_wrapper, hf_tokenizer = _resolve_hf_tokenizer(tokenizer)
    if requires_chat_template:
        _validate_chat_compatibility(hf_tokenizer)
    gigatoken = _load_gigatoken()
    try:
        encoding_backend = gigatoken.Tokenizer(hf_tokenizer).as_hf()
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"GigaToken does not support this Hugging Face tokenizer: {type(hf_tokenizer).__name__}."
        ) from error

    tokenizer_wrapper.tokenizer = _HuggingFaceTokenizerProxy(hf_tokenizer, encoding_backend)
    try:
        yield None
    finally:
        tokenizer_wrapper.tokenizer = hf_tokenizer
