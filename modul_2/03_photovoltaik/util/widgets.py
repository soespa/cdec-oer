"""Interaktive Widgets für Jupyter Notebooks."""

from IPython.display import display
from ipywidgets import FloatSlider, HBox, Label, VBox, interactive_output
from plotly.express import area
from pvlib.irradiance import get_total_irradiance


def interactive_irradiance(clear_sky, solar_position):


    label = Label()
    def calculate_irradiance(surface_tilt, surface_azimuth):
        irradiance = get_total_irradiance(
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            dni=clear_sky['dni'],
            ghi=clear_sky['ghi'],
            dhi=clear_sky['dhi'],
            solar_zenith=solar_position['zenith'],
            solar_azimuth=solar_position['azimuth']
        )
        total_energy = irradiance['poa_global'].sum() / 1000
        #rint('Bestrahlungsenergie:', total_energy.round(3), 'kWh/qm')

        label.value = f'Bestrahlungsenergie: {total_energy.round(3)} kWh/qm'

        fig = area(irradiance, y='poa_global', height=500)
        fig.update_yaxes(title='Bestrahlungsstärke [W/qm]', range=(0, 2000))
        fig.update_xaxes(title='Zeit')

        fig.show()

    tilt_slider = FloatSlider(value=0.0, min=0.0, max=90.0, step=0.5, description='Neigung')
    azimuth_slider = FloatSlider(value=0.0, min=0.0, max=360, step=0.5, description='Ausrichtung')


    out = interactive_output(calculate_irradiance, dict(surface_tilt=tilt_slider, surface_azimuth=azimuth_slider))

    display(HBox([
        VBox([tilt_slider, azimuth_slider, label]),
        out
    ]))
