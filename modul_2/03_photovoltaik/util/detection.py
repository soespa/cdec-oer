"""Erkennung von PV-Anlagen auf Orthophotos mit einem YOLO-Modell."""

import os
import pathlib

import cv2
import geopandas as gpd
import numpy as np
import plotly.express as px
import rasterio
from shapely import Polygon
from tqdm.auto import tqdm


class ObjectDetection:

    image_res = 640
    overlap = 320

    def __init__(self):

        base_path = pathlib.Path(__file__).parent.resolve()

        model_path = os.path.join(base_path, 'models', 'model.yaml')
        weight_path = os.path.join(base_path, 'models', 'weights', 'best.pt')

        # ultralytics (und damit PyTorch) wird erst hier geladen, da der Import einige Sekunden dauert
        from ultralytics import YOLO

        self.model = YOLO(model_path, verbose=True).load(weight_path)


    def predict(self, path):
        """
        Erkennt PV-Anlagen auf dem Orthophoto unter dem angegebenen Pfad.

        :param path: Pfad der JP2-Datei, z.B. 'data/orthophoto.jp2'
        :return: GeoDataFrame mit den erkannten PV-Anlagen
        """
        with rasterio.open(path) as src:
            return self._predict(src)


    def _predict(self, src: rasterio.DatasetReader):

        ds = src.read([1, 2, 3])

        rgb_image = np.stack(ds, axis=-1)

        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        width, height = bgr.shape[:2]

        res = self.image_res

        n = (width // res) * (height // res)

        progress = tqdm(total=n, desc='Detecting PV Modules')

        total_number = 0

        shapes = []

        resized = cv2.resize(rgb_image, (2500, 2500))

        fig = px.imshow(resized, width=1000, height=1000, binary_string=True)

        #fig = sp.make_subplots(rows=4, cols=4, horizontal_spacing=0.05, vertical_spacing=0.05)

        for i in range(0, width // res):
            for j in range(0, height // res):
                offset_x = j * res
                offset_y = i * res
                patch = bgr[i * res:(i + 1) * res, j * res:(j + 1) * res]

                #image_path = os.path.join('data', 'patches', f'tile_{i}_{j}.jpg')
                #cv2.imwrite(image_path, patch)

                results = self.model.predict(patch, imgsz=640, verbose=False)

                #results = self.model.predict(image_path, imgsz=640, verbose=True)

                result = results[0]
                boxes = result.boxes.xyxy.tolist()

                if len(boxes) > 0:

                    total_number += len(boxes)

                    for box in boxes:
                        x0, y0, x1, y1 = box

                        #fig.add_trace(go.Image(z=img, zmin=0, zmax=1, colormodel='rgb'), row=i + 1, col=j + 1)

                        x0 += offset_x
                        x1 += offset_x
                        y0 += offset_y
                        y1 += offset_y

                        fig.add_shape(
                            type='rect',
                            x0=x0 / 4,
                            y0=y0 / 4,
                            x1=x1 / 4,
                            y1=y1 / 4,
                            line=dict(
                                color='red'
                            )
                        )

                        polygon = Polygon([src.xy(y0, x0), src.xy(y0, x1), src.xy(y1, x1), src.xy(y1, x0)])

                        shapes.append(polygon)


                progress.update()

        fig.update_layout(width=1000, height=1000)
        fig.show()

        print(f'Anzahl gefundener PV-Anlagen: {total_number}')

        gdf = gpd.GeoDataFrame(geometry=shapes, crs=src.crs)

        return gdf
