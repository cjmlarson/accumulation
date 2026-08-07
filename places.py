"""Named geographic locations used in examples and plots."""

from typing import NamedTuple


class Place(NamedTuple):
    """A latitude-longitude coordinate."""

    lat: float
    lon: float


dresden = Place(51.0504, 13.7373)
wipkingerpark = Place(47.39263816580214, 8.521256406968515)
