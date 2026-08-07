"""Google Maps Static API helpers."""

import os
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

_STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"


def terrain(lat: float, lon: float, zoom: int, size: int = 640) -> str:
    """Return a Google Static Maps terrain URL."""
    parameters = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "maptype": "terrain",
        "key": os.environ["GOOGLE_MAPS_API_KEY"],
    }
    return f"{_STATIC_MAP_URL}?{urlencode(parameters)}"
