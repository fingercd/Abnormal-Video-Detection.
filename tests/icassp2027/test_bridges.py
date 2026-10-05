from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vadbench.token_reduction.bridges import BridgeUnsupportedError, create_observation_bridge


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.proj = nn.Linear(width * 3, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(self.qkv(value))


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = _Attention(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attn(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class _Encoder(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width), _Block(width)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


class _TupleBlock(nn.Module):
    def forward(self, value: torch.Tensor):
        return value + 1, torch.ones(value.shape[:2])


def _layers_model(path: tuple[str, ...], *, cls: bool = False) -> nn.Module:
    root = nn.Module()
    current = root
    for name in path[:-1]:
        child = nn.Module()
        setattr(current, name, child)
        current = child
    setattr(current, path[-1], nn.ModuleList([_Block(4), _Block(4)]))
    if cls:
        root.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
    return root


@pytest.mark.parametrize(
    ("encoder_id", "model", "expected_path"),
    [
        ("videomaev2", _Encoder(), "blocks"),
        (
            "timesformer",
            _layers_model(("timesformer", "encoder", "layer"), cls=True),
            "timesformer.encoder.layer",
        ),
        ("videomae", _layers_model(("encoder", "layer")), "encoder.layer"),
    ],
)
def test_active_bridges_resolve_only_known_loaded_block_paths(
    encoder_id: str, model: nn.Module, expected_path: str
) -> None:
    bridge = create_observation_bridge(encoder_id, model)
    receipt = bridge.receipt()

    assert receipt.probe_available
    assert receipt.block_path == expected_path
    assert receipt.block_count == 2
    assert not receipt.reduction_ready
    assert not receipt.reduction_available
    assert receipt.supports_grad is None  # module discovery is not a gradient test
    if encoder_id == "timesformer":
        assert receipt.layout == "unverified"  # no config/layout evidence in this fixture
        assert receipt.has_cls is True
        assert receipt.special_token_indices == (0,)
    elif encoder_id == "videomae":
        assert receipt.has_cls is False
        assert receipt.special_token_indices == ()
    else:
        assert receipt.has_cls is None
        assert receipt.special_token_indices is None


def test_vjepa_bridge_selects_vision_encoder_not_predictor_and_reports_no_grad() -> None:
    model = _layers_model(("vision_model", "encoder", "layer"))
    model.predictor = nn.Module()
    model.predictor.blocks = nn.ModuleList([_Block(4)])

    receipt = create_observation_bridge("vjepa2", model).receipt()

    assert receipt.block_path == "vision_model.encoder.layer"
    assert receipt.probe_available
    assert not receipt.supports_grad
    assert not receipt.reduction_ready


def test_hook_sites_are_real_modules_do_not_change_output_and_are_cleaned_up() -> None:
    model = _Encoder()
    bridge = create_observation_bridge("videomaev2", model)
    sites = bridge.observation_sites(depths=[0])
    assert set(sites) >= {
        "block.0.output",
        "block.0.norm1",
        "block.0.attn.output",
        "block.0.input",
        "block.0.mlp.pre_norm.input",
        "block.0.norm2",
        "block.0.mlp.pre_residual.output",
    }
    assert all(isinstance(module, nn.Module) for module in sites.values())

    value = torch.randn(2, 5, 4)
    baseline = model(value)
    observed: list[str] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: observed.append(name)
        )
        for name, module in sites.items()
    ]
    try:
        with_hooks = model(value)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(with_hooks, baseline)
    assert set(observed) == set(sites)
    assert all(not module._forward_hooks for module in sites.values())


def test_unknown_or_unverified_paths_fail_closed() -> None:
    with pytest.raises(BridgeUnsupportedError, match="未知 active encoder"):
        create_observation_bridge("unlisted", _Encoder())
    with pytest.raises(BridgeUnsupportedError, match="未在已加载模型上找到"):
        create_observation_bridge("videomae", _Encoder())
    bridge = create_observation_bridge("videomaev2", _Encoder())
    with pytest.raises(BridgeUnsupportedError, match="不存在 block depth"):
        bridge.observation_sites(depths=[4])


@pytest.mark.parametrize(
    ("encoder_id", "path"),
    [
        ("timesformer", ("timesformer", "encoder", "layer")),
        ("vjepa2", ("encoder", "layer")),
    ],
)
def test_identity_hook_api_preserves_native_block_tuple_fields(
    encoder_id: str, path: tuple[str, ...]
) -> None:
    root = nn.Module()
    current = root
    for name in path[:-1]:
        child = nn.Module()
        setattr(current, name, child)
        current = child
    block = _TupleBlock()
    setattr(current, path[-1], nn.ModuleList([block]))
    bridge = create_observation_bridge(encoder_id, root)
    native = block(torch.randn(2, 3, 4))
    hidden = bridge.block_output_tensor(0, native)
    replaced = bridge.replace_block_output_tensor(0, native, hidden)
    assert isinstance(replaced, tuple)
    assert replaced[0] is hidden and replaced[1] is native[1]
    with pytest.raises(BridgeUnsupportedError, match="shape"):
        bridge.replace_block_output_tensor(0, native, hidden[:, :-1])


def test_bridges_locate_real_transformers_classes_without_loading_weights():
    transformers = pytest.importorskip("transformers")
    vm = transformers.VideoMAEModel(
        transformers.VideoMAEConfig(
            image_size=16,
            patch_size=8,
            num_frames=4,
            tubelet_size=2,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=48,
        )
    )
    vm_bridge = create_observation_bridge("videomae", vm)
    assert vm_bridge.receipt().block_path == "encoder.layer"
    assert (
        vm_bridge.observation_sites([0])["block.0.mlp.pre_residual.output"]
        is vm.encoder.layer[0].output.dropout
    )
    ts = transformers.TimesformerModel(
        transformers.TimesformerConfig(
            image_size=16,
            patch_size=8,
            num_frames=4,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=48,
        )
    )
    ts_bridge = create_observation_bridge("timesformer", ts)
    assert ts_bridge.receipt().layout == "divided_space_time"
    sites = ts_bridge.observation_sites([0])
    assert (
        sites["block.0.temporal.attn.pre_projection.output"]
        is ts.encoder.layer[0].temporal_attention
    )
    assert sites["block.0.temporal.attn.projection.output"] is ts.encoder.layer[0].temporal_dense
    assert sites["block.0.spatial.attn.output"] is ts.encoder.layer[0].attention
    if hasattr(transformers, "VJEPA2Model"):
        vj = transformers.VJEPA2Model(
            transformers.VJEPA2Config(
                crop_size=16,
                patch_size=8,
                frames_per_clip=4,
                hidden_size=24,
                num_hidden_layers=2,
                num_attention_heads=2,
                pred_hidden_size=24,
                pred_num_hidden_layers=1,
                pred_num_attention_heads=2,
            )
        )
        vj_bridge = create_observation_bridge("vjepa2", vj)
        assert vj_bridge.receipt().block_path == "encoder.layer"
        assert not vj_bridge.receipt().supports_grad
        assert all("predictor" not in path for path in vj_bridge.observation_sites())
