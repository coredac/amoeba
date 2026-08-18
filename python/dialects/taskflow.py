from .._mlir_libs._TaskflowExtensionPybind11 import (
    taskflow as _taskflow_extension,
)
from ..ir import Context
from ._taskflow_ops_gen import *


def register_dialect(
    context: Context | None = None,
    load: bool = True,
) -> None:
    """Register and optionally load the Taskflow dialect."""

    if context is None:
        _taskflow_extension.register_dialect(load=load)
    else:
        _taskflow_extension.register_dialect(context, load)
