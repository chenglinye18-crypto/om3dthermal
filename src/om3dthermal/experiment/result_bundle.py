"""Compatibility imports for result bundle ownership moved to result/."""

from om3dthermal.result import RESULT_FILES, to_jsonable, write_result_bundle

__all__ = ["RESULT_FILES", "to_jsonable", "write_result_bundle"]
