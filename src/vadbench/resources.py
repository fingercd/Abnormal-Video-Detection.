"""Access immutable resources bundled with the installed VADBench wheel."""

from __future__ import annotations

from importlib import resources
from pathlib import Path, PurePosixPath, PureWindowsPath


def package_resource_path(relative: str) -> Path:
    """Return an existing bundled resource addressed from ``vadbench/resources``.

    Resource consumers must use repository-relative POSIX paths such as
    ``configs/encoders/videomaev2-base.yaml``. Explicit user paths remain a
    caller concern and must not be passed here.
    """

    if not isinstance(relative, str) or not relative:
        raise ValueError("资源路径必须是非空字符串")
    selected = PurePosixPath(relative)
    windows_selected = PureWindowsPath(relative)
    if (
        selected.is_absolute()
        or windows_selected.is_absolute()
        or windows_selected.drive
        or ".." in selected.parts
        or "." in selected.parts
    ):
        raise ValueError(f"资源路径必须位于包资源目录内：{relative!r}")
    target = resources.files("vadbench").joinpath("resources", *selected.parts)
    if not target.is_file():
        raise FileNotFoundError(f"未找到 wheel 内资源：{relative}")
    try:
        return Path(target)
    except TypeError as exc:  # pragma: no cover - zip imports have no stable Path
        raise RuntimeError(f"包资源不是可稳定访问的文件：{relative}") from exc


__all__ = ["package_resource_path"]
