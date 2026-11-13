"""Dachflächen aus dem Solarkataster NRW (LANUK) herunterladen und aufbereiten."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import requests
from tqdm.auto import tqdm

from ._tiles import DATA_DIR, find_tile

# Archiv mit den Rohdaten (File Geodatabase des Solarkatasters NRW)
SOLARKATASTER_URL = 'https://uni-paderborn.sciebo.de/s/ykIixPbmVMpEwIr/download'
GDB_NAME = 'solarkataster.gdb'

# Spalten, die für die Analyse benötigt werden
COLUMNS = ['geb_id', 'richtung', 'neigung', 'dachtyp', 'himmel_kat', 'geometry']


def _download_solarkataster(data_dir):
    """Lädt die Rohdaten herunter (nur beim ersten Mal) und gibt den Pfad zur Geodatabase zurück."""
    gdb_path = data_dir / GDB_NAME

    if gdb_path.exists():
        return gdb_path

    data_dir.mkdir(parents=True, exist_ok=True)

    # Archiv in eine temporäre Datei schreiben (nicht in den Arbeitsspeicher)
    with tempfile.NamedTemporaryFile(suffix='.zip', dir=data_dir, delete=False) as tmp:
        zip_path = Path(tmp.name)

    partial_path = data_dir / (GDB_NAME + '.part')

    try:
        with requests.get(SOLARKATASTER_URL, stream=True) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0)) or None

            with open(zip_path, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True,
                desc='Lade Solarkataster herunter. Das kann einige Minuten dauern.'
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    progress.update(len(chunk))

        # Nur die Geodatabase des Solarkatasters entpacken
        if partial_path.exists():
            shutil.rmtree(partial_path)

        found = False
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                parts = Path(info.filename).parts
                if GDB_NAME not in parts:
                    continue

                relative = parts[parts.index(GDB_NAME) + 1:]
                if '..' in relative:
                    continue

                found = True
                target = partial_path.joinpath(*relative)

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, open(target, 'wb') as destination:
                        shutil.copyfileobj(source, destination)

        if not found:
            raise FileNotFoundError(f'Das heruntergeladene Archiv enthält keine {GDB_NAME}.')

        partial_path.rename(gdb_path)

    finally:
        zip_path.unlink(missing_ok=True)
        if partial_path.exists():
            shutil.rmtree(partial_path, ignore_errors=True)

    return gdb_path


def download_roofs(latitude, longitude, path):
    """
    Lädt die Dachflächen (Solarkataster NRW) für die Umgebung der Koordinaten herunter,
    bereitet die Daten auf und speichert sie als GeoJSON-Datei unter dem angegebenen Pfad.

    Die Datei kann anschließend mit `geopandas.read_file(path)` eingelesen werden.

    :param latitude: Breitengrad (WGS84)
    :param longitude: Längengrad (WGS84)
    :param path: Pfad der GeoJSON-Datei, z.B. 'data/dachflaechen.geojson'
    """
    path = Path(path)

    # 1 x 1 km Kachel, in der der Standort liegt
    tile = find_tile(latitude, longitude, 'Für diese Koordinaten konnten keine Dachkatasterdaten geladen werden.')

    # Rohdaten (nur beim ersten Mal) in den Ordner `data` herunterladen
    gdb_path = _download_solarkataster(DATA_DIR)

    # Nur die Dachflächen innerhalb der Kachel einlesen
    gdf = gpd.read_file(gdb_path, mask=tile.geometry)

    if gdf.empty:
        raise ValueError('Für diesen Standort wurden keine Dachflächen gefunden.')

    # Spalten filtern
    gdf = gdf[COLUMNS]

    # Index
    gdf.index.name = 'id'

    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver='GeoJSON')

    print(f'{len(gdf)} Dachflächen gespeichert in: {path}')
