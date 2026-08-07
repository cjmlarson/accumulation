"""Google Maps Static API helpers."""

import os

import googlemaps
from dotenv import load_dotenv

load_dotenv()

_CLIENT = googlemaps.Client(key=os.environ["GOOGLE_MAPS_API_KEY"])


def terrain(lat: float, lon: float, zoom: int, size: int = 640) -> bytes:
    """Return a Google Static Maps terrain image."""
    chunks = _CLIENT.static_map(
        center=(lat, lon),
        zoom=zoom,
        size=(size, size),
        format="png",
        maptype="terrain",
    )
    return b"".join(chunks)
