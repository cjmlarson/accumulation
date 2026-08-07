"""JRC flood-depth queries."""

from collections import OrderedDict, deque
from contextlib import ExitStack
from math import ceil, floor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy

_DATA_DIR = Path(__file__).parent / "data" / "jrc"
_NEIGHBORS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_Pixel = tuple[str, int, int]


def _coordinate(value: int, positive: str, negative: str) -> str:
    direction = negative if value < 0 else positive
    return f"{direction}{abs(value)}"


def tile(lat: float, lon: float) -> str:
    """Return the JRC tile containing a coordinate.

    JRC filenames identify a tile by its upper-left corner, so 15 N, 25 E
    maps to ``N20_E20``. Tile edges are not always exact integer coordinates,
    so adjacent tiles are checked against their actual GeoTIFF bounds.
    """
    north = ceil(lat / 10) * 10
    west = floor(lon / 10) * 10
    directory = _DATA_DIR / "RP10"

    for latitude in (north, north - 10, north + 10):
        for longitude in (west, west - 10, west + 10):
            latitude_token = _coordinate(latitude, "N", "S")
            longitude_token = _coordinate(longitude, "E", "W") if longitude else "W0"
            pattern = f"ID*_{latitude_token}_{longitude_token}_RP10_depth.tif"
            for path in directory.glob(pattern):
                with rasterio.open(path) as src:
                    bounds = src.bounds
                if (
                    bounds.left <= lon <= bounds.right
                    and bounds.bottom <= lat <= bounds.top
                ):
                    return path.name.split("_RP", 1)[0]

    raise FileNotFoundError(f"no JRC tile covers ({lat}, {lon})")


def _rp_path(tile: str, rp: int) -> Path:
    return _DATA_DIR / f"RP{rp}" / f"{tile}_RP{rp}_depth.tif"


def _permanent_path(tile: str) -> Path:
    return _DATA_DIR / "Permanent_WaterBodies" / f"{tile}_permanent_water.tif"


class _FloodRasters:
    def __init__(self, rp: int):
        self.rp = rp
        self._stack = ExitStack()
        self._rasters: dict[str, rasterio.io.DatasetReader] = {}
        self._blocks = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._stack.close()

    def raster(self, tile_name: str) -> rasterio.io.DatasetReader:
        if tile_name not in self._rasters:
            self._rasters[tile_name] = self._stack.enter_context(
                rasterio.open(_rp_path(tile_name, self.rp))
            )
        return self._rasters[tile_name]

    def block(self, tile_name: str, block_row: int, block_col: int):
        key = tile_name, block_row, block_col
        if key in self._blocks:
            self._blocks.move_to_end(key)
            return self._blocks[key]

        src = self.raster(tile_name)
        window = src.block_window(1, block_row, block_col)
        self._blocks[key] = src.read(1, window=window, masked=True), window
        if len(self._blocks) > 128:
            self._blocks.popitem(last=False)
        return self._blocks[key]

    def pixel(self, lat: float, lon: float) -> _Pixel:
        tile_name = tile(lat, lon)
        row, col = self.raster(tile_name).index(lon, lat)
        return tile_name, row, col

    def wet(self, pixel: _Pixel) -> bool:
        tile_name, row, col = pixel
        src = self.raster(tile_name)
        block_height, block_width = src.block_shapes[0]
        values, window = self.block(tile_name, row // block_height, col // block_width)
        value = values[row - int(window.row_off), col - int(window.col_off)]
        return not np.ma.is_masked(value) and value > 0

    def neighbor(self, pixel: _Pixel, offset: tuple[int, int]) -> _Pixel:
        tile_name, row, col = pixel
        row += offset[0]
        col += offset[1]
        src = self.raster(tile_name)

        if 0 <= row < src.height and 0 <= col < src.width:
            return tile_name, row, col

        lon, lat = xy(src.transform, row, col)
        return self.pixel(lat, lon)


def depth(lat: float, lon: float, rp: int) -> float:
    r"""Return flood depth in metres for a coordinate and return period.

    .. math::

        h = h(\mathrm{lat}, \mathrm{lon}, \mathrm{RP})

    where :math:`\mathrm{RP}` is the return period in years.

    JRC represents dry land as raster nodata, which is returned as zero.
    """
    path = _rp_path(tile(lat, lon), rp)
    with rasterio.open(path) as src:
        sample = next(src.sample([(lon, lat)], masked=True))
    return 0.0 if sample.mask[0] else float(sample[0])


def connected(lat: float, lon: float, rp: int) -> set[tuple[str, int, int]]:
    """Return the 8-connected wet pixels containing a coordinate.

    The connected component is found with an iterative flood-fill algorithm.
    Pixels are returned as ``(tile, row, column)`` tuples. If the coordinate
    is dry, the result is empty.
    """
    with _FloodRasters(rp) as rasters:
        start = rasters.pixel(lat, lon)
        if not rasters.wet(start):
            return set()

        found = {start}
        queue = deque([start])

        while queue:
            pixel = queue.popleft()
            for offset in _NEIGHBORS:
                candidate = rasters.neighbor(pixel, offset)
                if candidate not in found and rasters.wet(candidate):
                    found.add(candidate)
                    queue.append(candidate)

        return found


def floodplain(
    lat: float, lon: float, rp: int, bridge_m: float = 250
) -> set[tuple[str, int, int]]:
    """Return flood pixels joined across nearby permanent water.

    ``bridge_m`` limits how far a connection may cross permanent water, so it
    cannot follow a river arbitrarily upstream or downstream.
    """
    raise NotImplementedError
