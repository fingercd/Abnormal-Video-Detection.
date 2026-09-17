from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import vadbench.integrations.catalog as catalog_module
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG
from vadbench.integrations.catalog import (
    IntegrationCatalogError,
    default_catalog_path,
    load_default_integration_catalog,
    load_integration_catalog,
    register_catalog_integrations,
)
from vadbench.orchestration import BUILTIN_ENCODER_CONFIGS, load_encoder_definition
from vadbench.registry import ENCODER_REGISTRY, EncoderRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "registry" / "encoder-integrations.yaml"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "encoder-integration-catalog-v1.schema.json"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

EXPECTED_IDS = {
    "r2plus1d_18",
    "x3d",
    "mvitv2",
    "slowfast",
    "c3d",
    "i3d",
    "timesformer",
    "video_swin",
    "videomae",
    "videomaev2",
    "internvideo2",
    "videomamba",
    "vjepa2",
    "longvu",
    "videochat",
    "videochat_online",
    "videochat_flash",
    "ma_lmm",
    "moviechat",
    "streaming_vlm",
    "hermes_llava_ov",
}

EXPECTED_FIXED_SMOKE_PROFILES = {
    "c3d": ("c3d-16x112", 16, 2, 112),
    "r2plus1d_18": ("r2plus1d-16x112", 16, 2, 112),
    "mvitv2": ("mvitv2-16x224", 16, 2, 224),
    "i3d": ("i3d-8x256", 8, 8, 256),
    "x3d": ("x3d-s-13x182", 13, 6, 182),
    "slowfast": ("slowfast-32x256", 32, 2, 256),
    "timesformer": ("timesformer-8x224", 8, 2, 224),
    "videomae": ("videomae-16x224", 16, 2, 224),
    "video_swin": ("video-swin-32x224", 32, 2, 224),
}


def _raw_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def _write_catalog(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "encoder-integrations.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def test_catalog_contains_only_the_21_runtime_targets() -> None:
    catalog = load_integration_catalog(CATALOG_PATH, project_root=PROJECT_ROOT)

    assert len(catalog) == 21
    assert len(catalog.ids) == len(set(catalog.ids))
    assert set(catalog.ids) == EXPECTED_IDS
    assert set(BUILTIN_ENCODER_CONFIGS) == EXPECTED_IDS
    assert set(ENCODER_REGISTRY.names()) == EXPECTED_IDS
    assert all(
        record.status in {"planned", "integrated", "blocked"} for record in catalog.integrations
    )
    assert all(
        record.checkpoint.status == "verified"
        for record in catalog.integrations
        if record.status == "integrated"
    )

    definitions = [record.definition for record in catalog.integrations]
    locks = [record.upstream_lock for record in catalog.integrations]
    checkpoints = [record.checkpoint.registry_id for record in catalog.integrations]
    assert len(definitions) == len(set(definitions))
    assert len(locks) == len(set(locks))
    assert len(checkpoints) == len(set(checkpoints))


def test_catalog_matches_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    data = _raw_catalog()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(data)


def test_wheel_force_includes_all_catalog_resources() -> None:
    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
    assert "[tool.hatch.build.targets.wheel.force-include]" in pyproject
    for source in ("configs", "registry", "schemas", "integrations"):
        assert f'"{source}" = "vadbench/resources/{source}"' in pyproject


def test_default_catalog_prefers_source_checkout(monkeypatch) -> None:
    def fail_if_called(relative: str) -> Path:
        raise AssertionError(f"checkout catalog exists; package resource {relative!r} was called")

    monkeypatch.setattr(catalog_module, "package_resource_path", fail_if_called)
    assert default_catalog_path(PROJECT_ROOT) == CATALOG_PATH.resolve()


def test_default_catalog_falls_back_to_packaged_resource(tmp_path: Path, monkeypatch) -> None:
    resource_root = tmp_path / "installed" / "vadbench" / "resources"
    for source in ("configs", "registry", "integrations"):
        shutil.copytree(PROJECT_ROOT / source, resource_root / source)
    packaged_catalog = resource_root / "registry" / "encoder-integrations.yaml"

    def fake_package_resource(relative: str) -> Path:
        assert relative == "registry/encoder-integrations.yaml"
        return packaged_catalog

    monkeypatch.setattr(catalog_module, "package_resource_path", fake_package_resource)
    missing_checkout = tmp_path / "checkout-without-registry"

    assert default_catalog_path(missing_checkout) == packaged_catalog.resolve()
    catalog = load_default_integration_catalog(missing_checkout)
    assert catalog.source_path == packaged_catalog.resolve()
    assert len(catalog) == 21
    assert set(catalog.ids) == EXPECTED_IDS


def test_schema_and_loader_require_two_streaming_chunks(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    data = _raw_catalog()
    streaming = next(item for item in data["integrations"] if item["id"] == "videochat_online")
    streaming["smoke_profile"]["chunks"] = 1

    assert not jsonschema.Draft202012Validator(schema).is_valid(data)
    path = _write_catalog(tmp_path, data)
    with pytest.raises(IntegrationCatalogError, match="至少需要 2 个 chunks"):
        load_integration_catalog(path, project_root=PROJECT_ROOT)


@pytest.mark.parametrize("status", ["smoke_pass", "failed", "preflight_pass", "acquiring"])
def test_schema_and_loader_reject_run_history_as_catalog_status(
    tmp_path: Path,
    status: str,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    data = _raw_catalog()
    data["integrations"][0]["status"] = status

    assert not jsonschema.Draft202012Validator(schema).is_valid(data)
    with pytest.raises(IntegrationCatalogError, match="不在"):
        load_integration_catalog(_write_catalog(tmp_path, data), project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("definition", "configs//encoders/bad.yaml", "不可越界"),
        ("adapter_target", "vadbench..integrations:Adapter", "module:attribute"),
    ],
)
def test_schema_and_loader_share_path_and_target_syntax(
    tmp_path: Path,
    field: str,
    invalid_value: str,
    message: str,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    data = _raw_catalog()
    data["integrations"][0][field] = invalid_value

    assert not jsonschema.Draft202012Validator(schema).is_valid(data)
    path = _write_catalog(tmp_path, data)
    with pytest.raises(IntegrationCatalogError, match=message):
        load_integration_catalog(path, project_root=PROJECT_ROOT)


def test_fixed_family_routes_and_smoke_profiles_are_pinned() -> None:
    video_swin = DEFAULT_INTEGRATION_CATALOG.get("video_swin")
    assert video_swin.adapter_target == (
        "vadbench.integrations.torchvision_video:TorchvisionVideoAdapter"
    )
    assert video_swin.backend == "torchvision"
    assert video_swin.environment.runtime == "in_process"
    assert video_swin.environment.profile == "classic-video-v2"
    assert video_swin.feature_stage == "backbone_tokens"

    actual_profiles = {}
    for integration_id in EXPECTED_FIXED_SMOKE_PROFILES:
        smoke = DEFAULT_INTEGRATION_CATALOG.get(integration_id).smoke_profile
        actual_profiles[integration_id] = (
            smoke.profile,
            smoke.clip_frames,
            smoke.frame_stride,
            smoke.image_size,
        )
        assert smoke.chunks == 1

    assert actual_profiles == EXPECTED_FIXED_SMOKE_PROFILES
    assert len({values[0] for values in actual_profiles.values()}) == len(actual_profiles)


def test_existing_integrations_keep_targets_capabilities_and_references() -> None:
    video = DEFAULT_INTEGRATION_CATALOG.get("videomaev2")
    hermes = DEFAULT_INTEGRATION_CATALOG.get("hermes_llava_ov")

    assert video.adapter_target == "vadbench.integrations.videomaev2:VideoMAEv2Adapter"
    assert hermes.adapter_target == "vadbench.integrations.hermes:HermesLlavaOVAdapter"
    assert video.definition == "configs/encoders/videomaev2-base.yaml"
    assert hermes.definition == "configs/encoders/hermes-llava-ov-0.5b.yaml"
    assert video.upstream_lock == "integrations/videomaev2/upstream.lock.yaml"
    assert hermes.upstream_lock == "integrations/hermes/upstream.lock.yaml"
    assert video.checkpoint.registry_id == "videomaev2-base-hf"
    assert hermes.checkpoint.registry_id == "hermes-llava-ov-0.5b"

    assert ENCODER_REGISTRY.get_spec("videomaev2").capabilities == video.capabilities
    hermes_spec = ENCODER_REGISTRY.get_spec("hermes_llava_ov")
    assert hermes_spec.capabilities == hermes.capabilities
    assert hermes_spec.metadata["cache_owner"] == "language_model_decoder"


def test_all_targets_have_definitions_and_static_registration_status() -> None:
    catalog = DEFAULT_INTEGRATION_CATALOG
    for record in catalog.integrations:
        assert record.status in {"planned", "integrated", "blocked"}
        definition = load_encoder_definition(record.id, project_root=PROJECT_ROOT)
        assert definition["adapter"] == record.id
        assert (PROJECT_ROOT / record.definition).is_file()
        assert (PROJECT_ROOT / record.upstream_lock).is_file()


def test_registration_and_listing_do_not_import_heavy_model_dependencies(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] in {"torch", "torchvision", "transformers", "pytorchvideo"}:
            raise AssertionError(f"catalog listing imported heavy dependency {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    catalog = load_integration_catalog(CATALOG_PATH, project_root=PROJECT_ROOT)
    registry = EncoderRegistry()
    register_catalog_integrations(catalog, registry)
    register_catalog_integrations(catalog, registry)

    assert set(registry.names()) == EXPECTED_IDS
    assert all(spec.is_lazy for spec in registry.specs())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["integrations"].clear(), "integrations 不能为空"),
        (
            lambda data: data["integrations"][1].update({"id": data["integrations"][0]["id"]}),
            "id 重复",
        ),
        (
            lambda data: data["integrations"][0].update({"unexpected": True}),
            "未知字段",
        ),
        (
            lambda data: data["integrations"][0].update({"definition": "../outside.yaml"}),
            "不可越界",
        ),
    ],
)
def test_loader_rejects_structural_errors(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    data = _raw_catalog()
    mutation(data)
    path = _write_catalog(tmp_path, data)

    with pytest.raises(IntegrationCatalogError, match=message):
        load_integration_catalog(path, project_root=PROJECT_ROOT)


def test_integrated_target_must_have_real_definition_and_lock(tmp_path: Path) -> None:
    data = _raw_catalog()
    record = data["integrations"][0]
    record["status"] = "integrated"
    record["checkpoint"] = {"registry_id": "r2plus1d_18-default", "status": "registered"}
    record["definition"] = "configs/encoders/not-created.yaml"
    path = _write_catalog(tmp_path, data)

    with pytest.raises(IntegrationCatalogError, match="文件不存在"):
        load_integration_catalog(path, project_root=tmp_path)


def test_registration_rejects_preexisting_target_drift() -> None:
    registry = EncoderRegistry()
    record = DEFAULT_INTEGRATION_CATALOG.get("videomaev2")
    registry.register_lazy(
        record.id,
        "vadbench.integrations.videomaev2:WrongAdapter",
        capabilities=record.capabilities,
    )

    with pytest.raises(IntegrationCatalogError, match="不一致"):
        register_catalog_integrations(DEFAULT_INTEGRATION_CATALOG, registry)
