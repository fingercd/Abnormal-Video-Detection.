"""Verify a locally built VADBench wheel without importing the source checkout.

Usage:
    python scripts/verify_wheel_resources.py dist/vadbench-0.1.0-py3-none-any.whl
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_RESOURCES = (
    "configs/encoders/videomaev2-base.yaml",
    "registry/checkpoints.yaml",
    "schemas/video-manifest-v1.schema.json",
    "integrations/videomaev2/upstream.lock.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 wheel 包含运行时资源且不从源码导入")
    parser.add_argument("wheel", type=Path, help="已构建的 vadbench wheel")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise FileNotFoundError(f"wheel 不存在或不是 .whl：{wheel}")

    with tempfile.TemporaryDirectory(prefix="vadbench-wheel-check-") as temp_dir:
        temp = Path(temp_dir)
        site = temp / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(site),
                str(wheel),
            ],
            check=True,
            cwd=temp,
        )
        code = """
import os
from pathlib import Path
import vadbench
from vadbench.resources import package_resource_path
from vadbench.orchestration import load_encoder_definition, resolve_encoder_config
from vadbench.integrations.videomaev2_encoder import VideoMAEv2Encoder

site = Path(os.environ['VADBENCH_WHEEL_SITE']).resolve()
module = Path(vadbench.__file__).resolve()
assert module.is_relative_to(site), (module, site)
for relative in os.environ['VADBENCH_REQUIRED_RESOURCES'].split(os.pathsep):
    path = package_resource_path(relative)
    assert path.is_file(), path
    assert path.read_bytes(), path
definition = load_encoder_definition('videomaev2', project_root=Path.cwd())
assert definition['checkpoint']['constructor_key'] == 'model_name'
resolved = resolve_encoder_config({'encoder': {'adapter': 'videomaev2', 'definition': 'configs/encoders/videomaev2-base.yaml'}}, project_root=Path.cwd())
assert Path(resolved['constructor']['model_name']).is_absolute()
assert 'lab_anomaly' not in __import__('sys').modules
print('wheel resource import verified:', module)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site)
        environment["VADBENCH_WHEEL_SITE"] = str(site)
        environment["VADBENCH_REQUIRED_RESOURCES"] = os.pathsep.join(REQUIRED_RESOURCES)
        subprocess.run([sys.executable, "-c", code], check=True, cwd=temp, env=environment)


if __name__ == "__main__":
    main()
