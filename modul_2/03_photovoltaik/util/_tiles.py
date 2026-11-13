"""Kachelindex der NRW-Orthophotos (Kacheln mit 1 x 1 km).

Wird sowohl für den Orthophoto-Download als auch für die Auswahl der Dachflächen
(Solarkataster) verwendet.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyproj
from shapely.geometry import Point, Polygon

# Heruntergeladene Daten liegen im Ordner `data` neben dem Notebook
# (relativ zum aktuellen Arbeitsverzeichnis).
DATA_DIR = Path('data')

TILE_INDEX = Path(__file__).parent / 'data' / 'dop_nrw.csv'
TILE_SIZE = 1000  # Kantenlänge einer Kachel in Metern


def find_tile(latitude, longitude, not_found_message):
    """
    Bestimmt die Kachel, in der die Koordinaten (WGS84) liegen.

    :param latitude: Breitengrad
    :param longitude: Längengrad
    :param not_found_message: Fehlermeldung, falls die Koordinaten nicht in NRW liegen
    :return: Zeile des Kachelindex (u.a. `Kachelname` und `geometry` in EPSG:25832)
    """
    transformer = pyproj.Transformer.from_crs('WGS84', 'EPSG:25832', always_xy=True)
    x, y = transformer.transform(longitude, latitude)
    point = Point(x, y)

    meta = pd.read_csv(TILE_INDEX, skiprows=5, sep=';')

    # Polygon aus Eckpunkten erzeugen (Koordinatenursprung = untere linke Ecke)
    geometry = [
        Polygon([
            [east, north],
            [east + TILE_SIZE, north],
            [east + TILE_SIZE, north + TILE_SIZE],
            [east, north + TILE_SIZE],
        ]) for east, north in zip(meta['Koordinatenursprung_East'], meta['Koordinatenursprung_North'])
    ]

    tiles = gpd.GeoDataFrame(meta, geometry=geometry, crs='EPSG:25832')

    selection = tiles[tiles.contains(point)]
    if len(selection) == 0:
        raise ValueError(
            f'{not_found_message} '
            'Bitte stellen Sie sicher, dass sich die Koordinaten innerhalb von NRW befinden.'
        )

    return selection.iloc[0]
