"""VADBench：视频异常检测编码器、训练、评测与缓存实验框架。"""

from __future__ import annotations

__version__ = "0.1.0"


def _register_builtin_encoders() -> None:
    # 这里只注册 module:attribute 字符串，不会导入 torch/transformers 或加载权重。
    from vadbench.integrations import register_builtin_integrations

    register_builtin_integrations()


_register_builtin_encoders()
del _register_builtin_encoders
