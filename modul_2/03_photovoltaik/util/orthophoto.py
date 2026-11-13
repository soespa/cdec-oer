"""Orthophotos (Luftbilder) von opengeodata.nrw.de herunterladen und anzeigen."""

from pathlib import Path

import cv2
import numpy as np
import plotly.express as px
import rasterio as rio
import requests
from tqdm.auto import tqdm

from ._tiles import find_tile

BASE_URL = 'https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/'


def _contains_tile(path, tile_geometry):
    """Prüft, ob unter `path` bereits das (lesbare) Orthophoto dieser Kachel liegt."""
    if not path.exists():
        return False

    try:
        with rio.open(path) as src:
            left, bottom, right, top = src.bounds
    except rio.errors.RasterioIOError:
        return False

    center = tile_geometry.centroid
    return left <= center.x <= right and bottom <= center.y <= top


def download_orthophoto(coords, path):
    """
    Lädt das Orthophoto (1 x 1 km) für die Koordinaten herunter und speichert es unter dem angegebenen Pfad.

    Liegt unter dem Pfad bereits das Orthophoto für diese Koordinaten, wird es nicht erneut heruntergeladen.
    Gehört die vorhandene Datei zu einem anderen Ort, wird sie überschrieben.

    :param coords: Tuple (latitude, longitude) in WGS84
    :param path: Pfad der JP2-Datei, z.B. 'data/orthophoto.jp2'
    """

    latitude, longitude = coords
    path = Path(path)

    tile = find_tile(latitude, longitude, 'Für diese Koordinaten konnte kein Orthophoto gefunden werden.')
    tile_name = tile['Kachelname']

    if _contains_tile(path, tile.geometry):
        print(f'Orthophoto bereits vorhanden: {path}')
        return

    url = BASE_URL + tile_name + '.jp2'

    print(url)

    # Gegebenenfalls Ordner anlegen, sofern dieser noch nicht existiert.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Zunächst in eine temporäre Datei schreiben, damit bei einem Abbruch
    # keine unvollständige Datei zurückbleibt.
    partial_path = path.with_name(path.name + '.part')

    # Webanfrage
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    chunk_size = 1024

    # Photo als JP2 abspeichern
    with open(partial_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size // chunk_size, unit='KB', desc='Downloading image'):
            if chunk:
                f.write(chunk)

    partial_path.replace(path)

    print(f'Orthophoto gespeichert in: {path}')


def preview_orthophoto(path):
    """
    Zeigt eine Vorschau (2500 x 2500 px) des Orthophotos an.

    :param path: Pfad der JP2-Datei
    """

    with rio.open(path) as src:
        ds = src.read([1, 2, 3])

    rgb_image = np.stack(ds, axis=-1)

    resized = cv2.resize(rgb_image, (2500, 2500))

    fig = px.imshow(resized, title='Preview', width=1000, height=1000, binary_string=True)

    fig.show()
