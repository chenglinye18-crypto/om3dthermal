"""Lightweight byte-accurate local KV page/slot allocation truth."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVSlotRange:
    allocation_id: str
    die_id: int
    page_id: int
    offset_bytes: int
    size_bytes: int


class KVPhysicalAllocator:
    """First-fit packed page ranges with deterministic die ownership.

    This is an allocation/accounting abstraction only.  It intentionally has
    no row, bank, timing, or replacement-policy model.
    """

    def __init__(self, *, die_count: int, pages_per_die: int,
                 page_size_bytes: int) -> None:
        if min(die_count, pages_per_die, page_size_bytes) <= 0:
            raise ValueError("allocator dimensions must be positive")
        self.die_count = die_count
        self.pages_per_die = pages_per_die
        self.page_size_bytes = page_size_bytes
        capacity = pages_per_die * page_size_bytes
        self._free: list[list[tuple[int, int]]] = [
            [(0, capacity)] for _ in range(die_count)]
        self._live: dict[str, tuple[KVSlotRange, ...]] = {}

    @property
    def live_allocations(self) -> dict[str, tuple[KVSlotRange, ...]]:
        return dict(self._live)

    @property
    def allocated_bytes_per_die(self) -> tuple[int, ...]:
        capacity = self.pages_per_die * self.page_size_bytes
        return tuple(capacity - sum(end-start for start, end in free)
                     for free in self._free)

    def _reserve(self, allocation_id: str, die_id: int,
                 size_bytes: int) -> tuple[KVSlotRange, ...]:
        if allocation_id in self._live:
            raise ValueError(f"live allocation already exists: {allocation_id}")
        if not 0 <= die_id < self.die_count:
            raise ValueError("die owner is out of range")
        if size_bytes <= 0:
            raise ValueError("allocation size must be positive")
        available = sum(end-start for start, end in self._free[die_id])
        if available < size_bytes:
            raise MemoryError("local die capacity exhausted")
        remaining = size_bytes
        ranges: list[KVSlotRange] = []
        new_free: list[tuple[int, int]] = []
        for start, end in self._free[die_id]:
            if remaining == 0:
                new_free.append((start, end))
                continue
            take = min(remaining, end-start)
            cursor = start
            limit = start + take
            while cursor < limit:
                page_id, offset = divmod(cursor, self.page_size_bytes)
                piece = min(limit-cursor, self.page_size_bytes-offset)
                ranges.append(KVSlotRange(
                    allocation_id=allocation_id, die_id=die_id,
                    page_id=page_id, offset_bytes=offset, size_bytes=piece))
                cursor += piece
            if start + take < end:
                new_free.append((start+take, end))
            remaining -= take
        self._free[die_id] = new_free
        result = tuple(ranges)
        if sum(item.size_bytes for item in result) != size_bytes:
            raise RuntimeError("physical allocation bytes do not close")
        self._live[allocation_id] = result
        return result

    def allocate_vector(
        self, *, request_id: str, layer_id: int, token_id: int,
        kind: str, kv_head_id: int, vector_bytes: int,
        owners: tuple[int, ...],
    ) -> tuple[KVSlotRange, ...]:
        if kind not in {"K", "V"}:
            raise ValueError("KV vector kind must be K or V")
        if not owners:
            raise ValueError("whole-vector allocation requires owners")
        # K and V deliberately use the same owner selector for a paired head.
        owner = owners[(layer_id + token_id + kv_head_id) % len(owners)]
        allocation_id = (
            f"kv:{request_id}:l{layer_id}:t{token_id}:{kind}:h{kv_head_id}")
        return self._reserve(allocation_id, owner, vector_bytes)

    def allocate_bytes(self, allocation_id: str, *, die_id: int,
                       size_bytes: int) -> tuple[KVSlotRange, ...]:
        return self._reserve(allocation_id, die_id, size_bytes)

    def free(self, allocation_id: str) -> tuple[KVSlotRange, ...]:
        try:
            ranges = self._live.pop(allocation_id)
        except KeyError as error:
            raise ValueError(f"allocation is not live: {allocation_id}") from error
        by_die: dict[int, list[tuple[int, int]]] = {}
        for item in ranges:
            start = item.page_id * self.page_size_bytes + item.offset_bytes
            by_die.setdefault(item.die_id, []).append(
                (start, start + item.size_bytes))
        for die_id, released in by_die.items():
            intervals = sorted((*self._free[die_id], *released))
            merged: list[tuple[int, int]] = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            self._free[die_id] = merged
        return ranges

    def migrate_local_to_host(self, allocation_id: str, *,
                              retain_local: bool) -> int:
        if allocation_id not in self._live:
            raise ValueError("LOCAL_TO_HOST requires a live local allocation")
        size = sum(item.size_bytes for item in self._live[allocation_id])
        if not retain_local:
            self.free(allocation_id)
        return size

    def migrate_host_to_local(self, allocation_id: str, *, die_id: int,
                              size_bytes: int) -> tuple[KVSlotRange, ...]:
        return self.allocate_bytes(
            allocation_id, die_id=die_id, size_bytes=size_bytes)


def assert_no_live_overlap(
        allocations: dict[str, tuple[KVSlotRange, ...]],
        page_size_bytes: int) -> None:
    intervals: dict[int, list[tuple[int, int, str]]] = {}
    for allocation_id, ranges in allocations.items():
        for item in ranges:
            start = item.page_id * page_size_bytes + item.offset_bytes
            intervals.setdefault(item.die_id, []).append(
                (start, start + item.size_bytes, allocation_id))
    for die_ranges in intervals.values():
        ordered = sorted(die_ranges)
        for left, right in zip(ordered, ordered[1:]):
            if left[1] > right[0] and left[2] != right[2]:
                raise ValueError("overlapping live physical KV allocations")
