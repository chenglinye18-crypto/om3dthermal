"""Typed research-result serialization and bundle ownership."""

from .bundle import RESULT_FILES, write_result_bundle
from .serialization import to_jsonable

__all__ = ["RESULT_FILES", "to_jsonable", "write_result_bundle"]
