from typing import Any


def NumbaType(jit_cls) -> Any:
    return jit_cls.class_type.instance_type
