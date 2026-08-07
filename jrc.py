"""JRC flood-depth queries."""

from math import ceil, floor
from pathlib import Path

import rasterio

_DATA_DIR = Path(__file__).parent / "data" / "jrc"


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
