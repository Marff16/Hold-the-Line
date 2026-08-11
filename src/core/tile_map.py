"""Block-packing layout for procedural Hold The Line facilities.

Skyline-packs building-sized rectangles directly into a grid (see
``pack_blocks``); the gaps left between packed blocks are the implicit
streets. ``src.core.instances.create_packed_facility`` converts the packed
blocks into literal ``Rect`` obstacles for a ``FixedMapConfig``.

An earlier iteration of this module built an explicit tile grid (ground
textures, straight roads, rejection-sampled buildings with sprite art,
decorators) as a step toward this generator; that approach was abandoned in
favor of packing blocks directly, so it isn't here anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BuildingKind(Enum):
    HOUSE_WOOD = "house_wood"
    HOUSE_GRAY_SHINGLE = "house_gray_shingle"
    MANSION_RED_TILE = "mansion_red_tile"
    WAREHOUSE_CORRUGATED = "warehouse_corrugated"
    WAREHOUSE_LOADING_DOCK = "warehouse_loading_dock"
    INDUSTRIAL_ROOFTOP = "industrial_rooftop"
    INSTITUTION_DORMERED = "institution_dormered"


# (width, height) in cells at rotation 0, sized from each sprite's measured
# content aspect ratio (0.9-1.05 for the two house kinds, up to 2.16 for the
# dormered institution) and scaled up so buildings read clearly against the
# road grid. Purely a sizing reference for BUILDING_SIZED_BLOCK_POOL below -
# the game engine's buildings are plain Rect obstacles with no kind attached.
BUILDING_FOOTPRINT_CELLS: dict[BuildingKind, tuple[int, int]] = {
    BuildingKind.HOUSE_WOOD: (6, 6),
    BuildingKind.HOUSE_GRAY_SHINGLE: (7, 6),
    BuildingKind.MANSION_RED_TILE: (8, 6),
    BuildingKind.WAREHOUSE_CORRUGATED: (10, 6),
    BuildingKind.WAREHOUSE_LOADING_DOCK: (9, 6),
    BuildingKind.INDUSTRIAL_ROOFTOP: (7, 6),
    BuildingKind.INSTITUTION_DORMERED: (12, 6),
}

# Real building footprints (plus their 90-degree-swapped variant) as a
# pack_blocks() pool - for generators that want packed blocks to plausibly
# hold an actual building later, instead of DEFAULT_BLOCK_SIZES's generic mix.
BUILDING_SIZED_BLOCK_POOL: list[tuple[int, int]] = list(BUILDING_FOOTPRINT_CELLS.values()) + [
    (h, w) for w, h in BUILDING_FOOTPRINT_CELLS.values()
]


@dataclass(frozen=True)
class PackedBlock:
    origin: tuple[int, int]  # (row, col) of the block's top-left cell
    width: int
    height: int


# A block gets built directly from a random pool of predefined sizes, not
# tied to any particular BuildingKind - this step is just about the packing
# layout; matching a real building sprite to each block is a follow-up.
DEFAULT_BLOCK_SIZES: list[tuple[int, int]] = [
    (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4), (2, 4), (4, 2),
    (5, 3), (3, 5), (5, 4), (4, 5), (5, 5), (5, 2), (2, 5),
    (6, 3), (3, 6), (6, 4), (4, 6), (6, 5), (5, 6), (6, 6), (6, 2), (2, 6),
    (7, 4), (4, 7), (7, 5), (5, 7), (8, 4), (4, 8), (8, 5), (5, 8),
]


def pack_blocks(
    rng: np.random.Generator,
    cols: int,
    rows: int,
    footprint_pool: list[tuple[int, int]] = DEFAULT_BLOCK_SIZES,
    margin_rows: int = 5,
    margin_rows_top: int | None = None,
    margin_rows_bottom: int | None = None,
    gap_choices: tuple[int, ...] = (0, 2, 3, 4),
    max_attempts: int | None = None,
    max_consecutive_misses: int = 60,
) -> list[PackedBlock]:
    """Skyline-pack random rectangles from ``footprint_pool``.

    Tracks a per-column "skyline" - the current lowest free row in each
    column - rather than filling one shared row band at a time. For each
    randomly sampled rectangle, a random column is tried; it's placed at
    ``max(skyline[col:col+width])`` (however far down it has to sit to clear
    everything already above it), with a gap randomly picked from
    ``gap_choices`` cells added below before the skyline is raised for those
    columns. Because
    neighboring columns' skylines advance independently, block bottoms end
    up staggered across the width instead of lining up along shared row
    seams. The gaps left behind are the implicit streets - no road cells are
    painted by this function at all.

    ``margin_rows_top``/``margin_rows_bottom`` override ``margin_rows`` for
    just that side, for callers that need an asymmetric margin (e.g. a
    bigger reserved band at the bottom for a protected zone).
    """

    top = margin_rows if margin_rows_top is None else margin_rows_top
    bottom = margin_rows if margin_rows_bottom is None else margin_rows_bottom
    free_row_lo, free_row_hi = top, rows - bottom
    if free_row_hi <= free_row_lo:
        raise ValueError(f"margins leave no free rows for a grid.rows={rows} map (top={top}, bottom={bottom})")

    if max_attempts is None:
        max_attempts = cols * (free_row_hi - free_row_lo)

    skyline = [free_row_lo] * cols
    blocks: list[PackedBlock] = []
    attempts = 0
    consecutive_misses = 0

    while attempts < max_attempts and consecutive_misses < max_consecutive_misses:
        attempts += 1
        width, height = footprint_pool[int(rng.integers(len(footprint_pool)))]
        if width > cols:
            consecutive_misses += 1
            continue

        col = int(rng.integers(0, cols - width + 1))
        row = max(skyline[col : col + width])
        if row + height > free_row_hi:
            consecutive_misses += 1
            continue

        blocks.append(PackedBlock((row, col), width, height))
        gap = int(gap_choices[int(rng.integers(len(gap_choices)))])
        for c in range(col, col + width):
            skyline[c] = row + height + gap
        consecutive_misses = 0

    return blocks
