from types import SimpleNamespace

import pytest

from megatron.bridge.data.conversation_processing import resolve_chat_template_kwargs
from megatron.bridge.data.packing.gigatoken import packed_sft_tokenizer_backend


class _FakeHFTokenizer:
    is_fast = True
    chat_template = None

    def __init__(self, prefix: int) -> None:
        self.prefix = prefix

    def __call__(self, text: str, **kwargs):
        del kwargs
        return SimpleNamespace(input_ids=[self.prefix, len(text)])

    def encode(self, text: str, **kwargs):
        del kwargs
        return [self.prefix, len(text)]

    def tokenize(self, text: str, **kwargs):
        del kwargs
        return [str(self.prefix), text]

    def apply_chat_template(self, conversation, **kwargs):
        del kwargs
        return self(str(conversation))


def _make_megatron_tokenizer(base_tokenizer: _FakeHFTokenizer, *, library: str = "huggingface"):
    return SimpleNamespace(library=library, _tokenizer=SimpleNamespace(tokenizer=base_tokenizer))


def test_disabled_backend_preserves_original_tokenizer(monkeypatch):
    base_tokenizer = _FakeHFTokenizer(prefix=1)
    tokenizer = _make_megatron_tokenizer(base_tokenizer)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: pytest.fail("disabled backend must not import GigaToken"),
    )

    with packed_sft_tokenizer_backend(
        tokenizer,
        use_gigatoken=False,
        requires_chat_template=False,
    ):
        assert tokenizer._tokenizer.tokenizer is base_tokenizer

    assert tokenizer._tokenizer.tokenizer is base_tokenizer


def test_enabled_backend_routes_encoding_and_restores_original(monkeypatch):
    base_tokenizer = _FakeHFTokenizer(prefix=1)
    fast_tokenizer = _FakeHFTokenizer(prefix=9)
    tokenizer = _make_megatron_tokenizer(base_tokenizer)

    class FakeGigaToken:
        def __init__(self, source):
            assert source is base_tokenizer

        def as_hf(self):
            return fast_tokenizer

    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: SimpleNamespace(Tokenizer=FakeGigaToken),
    )

    with packed_sft_tokenizer_backend(
        tokenizer,
        use_gigatoken=True,
        requires_chat_template=False,
    ):
        active_tokenizer = tokenizer._tokenizer.tokenizer
        assert active_tokenizer("hello").input_ids == [9, 5]
        assert active_tokenizer.encode("hello") == [9, 5]
        assert active_tokenizer.tokenize("hello") == ["9", "hello"]

    assert tokenizer._tokenizer.tokenizer is base_tokenizer


def test_enabled_backend_preserves_chat_rendering_and_routes_encoding(monkeypatch):
    base_tokenizer = _FakeHFTokenizer(prefix=1)
    base_tokenizer.chat_template = "{% if preserve_thinking %}history{% endif %}"
    fast_tokenizer = _FakeHFTokenizer(prefix=9)
    tokenizer = _make_megatron_tokenizer(base_tokenizer)

    class FakeGigaToken:
        def __init__(self, source):
            assert source is base_tokenizer

        def as_hf(self):
            return fast_tokenizer

    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: SimpleNamespace(Tokenizer=FakeGigaToken),
    )
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._supports_standard_chat_template",
        lambda _: True,
    )

    with packed_sft_tokenizer_backend(
        tokenizer,
        use_gigatoken=True,
        requires_chat_template=True,
    ):
        active_tokenizer = tokenizer._tokenizer.tokenizer
        assert resolve_chat_template_kwargs(
            active_tokenizer,
            {"truncate_history_thinking": False},
        ) == {"preserve_thinking": True}
        assert active_tokenizer.apply_chat_template([{"role": "user", "content": "hello"}]).input_ids == [
            9,
            38,
        ]

    assert tokenizer._tokenizer.tokenizer is base_tokenizer


def test_unsupported_tokenizer_rejected_before_dependency_import(monkeypatch):
    tokenizer = _make_megatron_tokenizer(_FakeHFTokenizer(prefix=1), library="sentencepiece")
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: pytest.fail("unsupported tokenizers must fail before importing GigaToken"),
    )

    with pytest.raises(ValueError, match="supports only HuggingFaceTokenizer"):
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=False,
        ):
            pass


def test_custom_chat_template_method_rejected_before_dependency_import(monkeypatch):
    tokenizer = _make_megatron_tokenizer(_FakeHFTokenizer(prefix=1))
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: pytest.fail("custom chat tokenizers must fail before importing GigaToken"),
    )

    with pytest.raises(ValueError, match="custom apply_chat_template"):
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=True,
        ):
            pass


def test_instance_custom_chat_template_method_rejected_before_dependency_import(monkeypatch):
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    base_tokenizer = _FakeHFTokenizer(prefix=1)
    monkeypatch.setattr(
        PreTrainedTokenizerBase,
        "apply_chat_template",
        _FakeHFTokenizer.apply_chat_template,
    )
    base_tokenizer.apply_chat_template = lambda conversation, **kwargs: str((conversation, kwargs))
    tokenizer = _make_megatron_tokenizer(base_tokenizer)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: pytest.fail("instance-custom chat tokenizers must fail before importing GigaToken"),
    )

    with pytest.raises(ValueError, match="custom apply_chat_template"):
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=True,
        ):
            pass


def test_missing_dependency_has_actionable_error(monkeypatch):
    tokenizer = _make_megatron_tokenizer(_FakeHFTokenizer(prefix=1))

    def raise_missing_dependency(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr("megatron.bridge.data.packing.gigatoken.importlib.import_module", raise_missing_dependency)

    with pytest.raises(ModuleNotFoundError, match="optional `gigatoken` package"):
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=False,
        ):
            pass


def test_missing_gigatoken_transitive_dependency_is_not_misreported(monkeypatch):
    tokenizer = _make_megatron_tokenizer(_FakeHFTokenizer(prefix=1))

    def raise_missing_dependency(name):
        assert name == "gigatoken"
        raise ModuleNotFoundError(name="gigatoken_native")

    monkeypatch.setattr("megatron.bridge.data.packing.gigatoken.importlib.import_module", raise_missing_dependency)

    with pytest.raises(ModuleNotFoundError) as error:
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=False,
        ):
            pass

    assert error.value.name == "gigatoken_native"


@pytest.mark.parametrize(
    "chat_template",
    [
        "{% generation %}{{ message.content }}{% endgeneration %}",
        {"default": "{{ message.content }}", "tool_use": "{%- generation %}tool{% endgeneration %}"},
    ],
)
def test_generation_block_chat_template_is_rejected(monkeypatch, chat_template):
    base_tokenizer = _FakeHFTokenizer(prefix=1)
    base_tokenizer.chat_template = chat_template
    tokenizer = _make_megatron_tokenizer(base_tokenizer)
    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._supports_standard_chat_template",
        lambda _: True,
    )

    with pytest.raises(ValueError, match="does not return the offset mapping"):
        with packed_sft_tokenizer_backend(
            tokenizer,
            use_gigatoken=True,
            requires_chat_template=True,
        ):
            pass


def test_enabled_backend_allows_multiple_bridge_workers(monkeypatch):
    from megatron.bridge.data.packing.offline import tokenize_dataset

    base_tokenizer = _FakeHFTokenizer(prefix=1)
    fast_tokenizer = _FakeHFTokenizer(prefix=9)
    tokenizer = _make_megatron_tokenizer(base_tokenizer)

    class FakeGigaToken:
        def __init__(self, source):
            assert source is base_tokenizer

        def as_hf(self):
            return fast_tokenizer

    class EmptyDataset:
        tokenizer = SimpleNamespace(eod=0)
        pad_seq_length_to_mult = 1

    def materialize(dataset, num_workers):
        assert isinstance(dataset, EmptyDataset)
        assert num_workers == 2
        assert tokenizer._tokenizer.tokenizer("hello").input_ids == [9, 5]
        return []

    monkeypatch.setattr(
        "megatron.bridge.data.packing.gigatoken._load_gigatoken",
        lambda: SimpleNamespace(Tokenizer=FakeGigaToken),
    )
    monkeypatch.setattr("megatron.bridge.data.packing.offline._materialize_dataset_items", materialize)

    assert (
        tokenize_dataset(
            "unused.jsonl",
            tokenizer=tokenizer,
            max_seq_length=16,
            seed=123,
            num_tokenizer_workers=2,
            use_gigatoken=True,
            dataset_builder=lambda *args, **kwargs: EmptyDataset(),
        )
        == []
    )
    assert tokenizer._tokenizer.tokenizer is base_tokenizer
