"""Plots for JRC flood-depth data."""

from io import BytesIO

import matplotlib.pyplot as plt
import mercantile
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.transform import array_bounds
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform,
    transform_bounds,
)
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds

import google as google_maps
import jrc


def zoom(lat: float, lon: float, level: int, size: int = 640) -> tuple[float, ...]:
    """Return bounds for a centered Web Mercator map view."""
    x, y = mercantile.xy(lon, lat, truncate=True)
    radius = size * mercantile.CE / (2 * 256 * 2**level)
    lower_left = mercantile.lnglat(x - radius, y - radius)
    upper_right = mercantile.lnglat(x + radius, y + radius)
    return lower_left.lng, lower_left.lat, upper_right.lng, upper_right.lat


_zoom_bounds = zoom


def mercator(values, source_transform):
    """Reproject a categorical raster from WGS84 to Web Mercator."""
    height, width = values.shape
    bounds = array_bounds(height, width, source_transform)
    destination_transform, destination_width, destination_height = (
        calculate_default_transform("EPSG:4326", "EPSG:3857", width, height, *bounds)
    )
    projected = np.empty((destination_height, destination_width), dtype=values.dtype)
    reproject(
        values,
        projected,
        src_transform=source_transform,
        src_crs="EPSG:4326",
        dst_transform=destination_transform,
        dst_crs="EPSG:3857",
        resampling=Resampling.nearest,
    )
    return projected, destination_transform


def flooded(lat: float, lon: float, rp: int, zoom: int = 12) -> plt.Axes:
    """Plot wet and dry pixels around a coordinate.

    Zoom levels follow the standard 256-pixel Web Mercator map pyramid.
    """
    tile = jrc.tile(lat, lon)
    rp_path = jrc._rp_path(tile, rp)
    permanent_path = jrc._permanent_path(tile)
    size = 640
    view_bounds = _zoom_bounds(lat, lon, zoom, size)
    left, bottom, right, top = view_bounds

    with rasterio.open(rp_path) as src:
        left = max(src.bounds.left, left)
        right = min(src.bounds.right, right)
        bottom = max(src.bounds.bottom, bottom)
        top = min(src.bounds.top, top)
        window = from_bounds(left, bottom, right, top, src.transform)
        values = src.read(1, window=window)
        flooded = (values != src.nodata).astype("uint8")
        left, bottom, right, top = window_bounds(window, src.transform)
        source_transform = src.window_transform(window)

    with rasterio.open(permanent_path) as src:
        permanent = src.read(1, window=window)
    flooded[permanent == 1] = 2

    flooded, destination_transform = mercator(flooded, source_transform)
    left, bottom, right, top = array_bounds(*flooded.shape, destination_transform)
    extent = (left, right, bottom, top)
    point_x, point_y = transform("EPSG:4326", "EPSG:3857", [lon], [lat])

    terrain = plt.imread(
        BytesIO(google_maps.terrain(lat, lon, zoom, size)), format="png"
    )
    left, bottom, right, top = transform_bounds("EPSG:4326", "EPSG:3857", *view_bounds)
    terrain_extent = (left, right, bottom, top)

    _, ax = plt.subplots()
    ax.imshow(terrain, extent=terrain_extent, interpolation="nearest", origin="upper")
    ax.imshow(
        flooded,
        cmap=ListedColormap([(1, 1, 1, 0), (0.17, 0.51, 0.73, 0.7), (0, 0, 0, 0.8)]),
        extent=extent,
        interpolation="nearest",
        origin="upper",
    )
    ax.plot(point_x, point_y, marker="+", color="black")
    ax.set_title(f"JRC RP{rp} flooded area")
    ax.set_axis_off()
    return ax
