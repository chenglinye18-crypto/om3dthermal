"""Fixed-size pages for explicitly already-local-resident data objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .power.physical_capacity import PhysicalCapacityLayout


ResidentObjectType = Literal["WEIGHT", "KV", "OTHER"]
CANONICAL_RESIDENT_SET_STATUS = "NOT_YET_BOUND"


class ResidentCapacityExceededError(ValueError):
    """The caller-provided resident set cannot fit local physical slots."""


@dataclass(frozen=True)
class ResidentDataObject:
    object_id: str
    object_type: ResidentObjectType
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("resident object_id must be non-empty")
        if self.object_type not in {"WEIGHT", "KV", "OTHER"}:
            raise ValueError("resident object_type must be WEIGHT, KV, or OTHER")
        if isinstance(self.size_bytes, bool) or not isinstance(
                self.size_bytes, int):
            raise TypeError("resident object size_bytes must be an int")
        if self.size_bytes <= 0:
            raise ValueError("resident object size_bytes must be positive")


@dataclass(frozen=True)
class ResidentDataPage:
    page_id: str
    parent_object_id: str
    object_type: ResidentObjectType
    page_index: int
    size_bytes: int
    capacity_bytes: int


@dataclass(frozen=True)
class ResidentPageLayout:
    page_size_bytes: int
    objects: tuple[ResidentDataObject, ...]
    pages: tuple[ResidentDataPage, ...]
    object_count: int
    page_count: int
    logical_resident_bytes: int
    allocated_page_bytes: int
    internal_fragmentation_bytes: int
    internal_fragmentation_ratio: float
    physical_slot_count: int
    remaining_slot_count: int
    remaining_physical_capacity_bytes: int
    capacity_feasible: bool
    pages_by_object_type: dict[str, int]
    logical_bytes_by_object_type: dict[str, int]
    input_semantics: str
    page_sharing: bool
    page_to_physical_slot_mapping_included: bool
    host_capacity_included: bool
    ordering_semantics: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_resident_page_layout(
        objects: tuple[ResidentDataObject, ...],
        physical_layout: PhysicalCapacityLayout,
        ) -> ResidentPageLayout:
    """Decompose caller-declared local objects without placement or spill.

    Each page consumes one complete physical slot.  Object tails are never
    shared, and page ordering is input-object order followed by ascending
    page index.  The caller, not this function, owns residency decisions.
    """
    object_ids = tuple(obj.object_id for obj in objects)
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("resident object IDs must be unique")
    page_size = physical_layout.slot_capacity_bytes
    if page_size <= 0:
        raise ValueError("physical slot capacity must be positive")
    if physical_layout.total_capacity_bytes != (
            physical_layout.physical_slot_count * page_size):
        raise ValueError("physical slot count and total capacity do not close")
    if any(slot.capacity_bytes != page_size
           for slot in physical_layout.slot_classes):
        raise ValueError("resident pages require uniform physical slot capacity")

    pages: list[ResidentDataPage] = []
    pages_by_type = {name: 0 for name in ("WEIGHT", "KV", "OTHER")}
    bytes_by_type = {name: 0 for name in ("WEIGHT", "KV", "OTHER")}
    logical_bytes = 0
    for obj in objects:
        logical_bytes += obj.size_bytes
        bytes_by_type[obj.object_type] += obj.size_bytes
        page_count = (obj.size_bytes + page_size - 1) // page_size
        pages_by_type[obj.object_type] += page_count
        remaining = obj.size_bytes
        for page_index in range(page_count):
            logical_page_size = min(page_size, remaining)
            pages.append(ResidentDataPage(
                page_id=f"{obj.object_id}:page:{page_index}",
                parent_object_id=obj.object_id,
                object_type=obj.object_type,
                page_index=page_index,
                size_bytes=logical_page_size,
                capacity_bytes=page_size,
            ))
            remaining -= logical_page_size
        if remaining != 0:
            raise RuntimeError("resident object page decomposition did not close")

    page_tuple = tuple(pages)
    page_count = len(page_tuple)
    if page_count > physical_layout.physical_slot_count:
        raise ResidentCapacityExceededError(
            "resident data set exceeds available local physical slots")
    allocated_bytes = page_count * page_size
    fragmentation = allocated_bytes - logical_bytes
    remaining_slots = physical_layout.physical_slot_count - page_count
    return ResidentPageLayout(
        page_size_bytes=page_size,
        objects=objects,
        pages=page_tuple,
        object_count=len(objects),
        page_count=page_count,
        logical_resident_bytes=logical_bytes,
        allocated_page_bytes=allocated_bytes,
        internal_fragmentation_bytes=fragmentation,
        internal_fragmentation_ratio=(
            fragmentation / allocated_bytes if allocated_bytes else 0.0),
        physical_slot_count=physical_layout.physical_slot_count,
        remaining_slot_count=remaining_slots,
        remaining_physical_capacity_bytes=remaining_slots * page_size,
        capacity_feasible=True,
        pages_by_object_type=pages_by_type,
        logical_bytes_by_object_type=bytes_by_type,
        input_semantics="EXPLICIT_ALREADY_LOCAL_RESIDENT_OBJECTS_ONLY",
        page_sharing=False,
        page_to_physical_slot_mapping_included=False,
        host_capacity_included=False,
        ordering_semantics=(
            "INPUT_OBJECT_ORDER_THEN_ASCENDING_LOGICAL_PAGE_INDEX"),
    )
