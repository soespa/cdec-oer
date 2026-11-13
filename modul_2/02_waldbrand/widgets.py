"""
utils_regression.py

Utility functions for interactive linear regression exploration in Jupyter notebooks.

Public API:
- interactive_regression(X, y)
- animation_gradient_descent(X, y)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import (
    FloatSlider,
    Button,
    HBox,
    VBox,
    interactive_output,
    Label,
)
from sklearn.metrics import mean_squared_error

__all__ = ["interactive_regression", "animation_gradient_descent"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_1d_numpy(array_like) -> np.ndarray:
    """
    Convert input to a 1D NumPy array.

    Parameters
    ----------
    array_like : array-like
        Input array or pandas Series.

    Returns
    -------
    np.ndarray
        1D numpy array.
    """
    arr = np.asarray(array_like)
    return np.squeeze(arr)


def _regression_predictions(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Compute ŷ for a simple linear model ŷ = slope * x + intercept."""
    return slope * x + intercept


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def interactive_regression(X, y) -> None:
    """
    Interactive linear regression with sliders for slope and intercept.

    Displays:
    - Scatter plot of (x, y)
    - Regression line for chosen slope & intercept
    - MSE label (updated live)

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1)
        Feature values.
    y : array-like, shape (n,)
        Target values.

    Notes
    -----
    The axis titles are kept as in the original implementation.
    """
    x = _ensure_1d_numpy(X)
    y = _ensure_1d_numpy(y)

    # Initial parameters (kept as in original)
    initial_slope = 0.0
    initial_intercept = 10.0

    mse_label = Label()

    def update_plot(slope: float, intercept: float) -> None:
        y_pred = _regression_predictions(x, slope, intercept)
        mse = mean_squared_error(y, y_pred)

        # Scatter + regression line
        scatter = go.Scatter(x=x, y=y, mode="markers", name="Daten")
        regression_line = go.Scatter(x=x, y=y_pred, mode="lines", name="Regressionsgerade")

        layout = go.Layout(
            xaxis=dict(title="Oberflächentemperatur"),
            yaxis=dict(title="Anzahl der Waldbrände", range=[min(y), max(y)]),
            height=500,
            margin=dict(t=10, b=10, l=10, r=10),
        )

        fig = go.Figure(data=[scatter, regression_line], layout=layout)

        # Styling (kept behavior)
        fig.update_yaxes(
            range=(0, 1.1 * float(np.max(y))),
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=2,
            showgrid=False,
        )
        fig.update_xaxes(
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=2,
            showgrid=False,
        )

        mse_label.value = f"Mean Squared Error: {round(mse, 2)}"
        fig.show()

    # Sliders
    slope_slider = FloatSlider(
        value=initial_slope, min=-0.5, max=1.5, step=0.01, description="Steigung:"
    )
    intercept_slider = FloatSlider(
        value=initial_intercept, min=-5.0, max=30.0, step=0.1, description="Y-Achsenabschnitt:"
    )

    # Increment/decrement buttons
    slope_increment_btn = Button(description="+", layout={"width": "50px"})
    slope_decrement_btn = Button(description="-", layout={"width": "50px"})
    intercept_increment_btn = Button(description="+", layout={"width": "50px"})
    intercept_decrement_btn = Button(description="-", layout={"width": "50px"})

    def _bump(slider: FloatSlider, step: float) -> None:
        slider.value = round(slider.value + step, 2)

    slope_increment_btn.on_click(lambda _: _bump(slope_slider, 0.01))
    slope_decrement_btn.on_click(lambda _: _bump(slope_slider, -0.01))
    intercept_increment_btn.on_click(lambda _: _bump(intercept_slider, 0.1))
    intercept_decrement_btn.on_click(lambda _: _bump(intercept_slider, -0.1))

    # Layout
    controls = VBox(
        [
            VBox([slope_slider, HBox([slope_increment_btn, slope_decrement_btn])]),
            VBox([intercept_slider, HBox([intercept_increment_btn, intercept_decrement_btn])]),
            mse_label,
        ]
    )

    out = interactive_output(update_plot, {"slope": slope_slider, "intercept": intercept_slider})
    display(VBox([controls, out]))


def animation_gradient_descent(X, y) -> None:
    """
    Create an animated visualization of gradient descent for linear regression.

    The animation shows how slope and intercept evolve over time and updates
    a regression line and an annotation with current parameters and MSE.

    Parameters
    ----------
    X : array-like, shape (n,) or (n, 1)
        Feature values.
    y : array-like, shape (n,)
        Target values.

    Notes
    -----
    Default hyperparameters are kept identical to the original implementation.
    Axis titles are preserved.
    """
    # Hyperparameters (kept as original behavior)
    num_frames = 200
    subsample_stride = 50
    learning_rate = 0.001

    x = _ensure_1d_numpy(X)
    y = _ensure_1d_numpy(y)

    # Initialize parameters (kept as original)
    slope = -0.5
    intercept = 20.0

    def run_gradient_descent(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        slope0: float,
        intercept0: float,
        steps: int,
        lr: float,
    ) -> list[Tuple[float, float, float]]:
        """
        Perform batch gradient descent and collect (slope, intercept, MSE) per step.
        """
        s = float(slope0)
        b = float(intercept0)
        history: list[Tuple[float, float, float]] = []

        n = float(len(y_vals))
        for _ in range(steps):
            y_pred = _regression_predictions(x_vals, s, b)

            # Gradients for MSE loss
            grad_s = (-2.0 / n) * np.sum((y_vals - y_pred) * x_vals)
            grad_b = (-2.0 / n) * np.sum(y_vals - y_pred)

            # Parameter update
            s -= lr * grad_s
            b -= lr * grad_b

            history.append((s, b, mean_squared_error(y_vals, y_pred)))

        return history

    # Compute a dense trajectory and then sample frames
    total_steps = num_frames * subsample_stride
    trajectory = run_gradient_descent(
        x_vals=x,
        y_vals=y,
        slope0=slope,
        intercept0=intercept,
        steps=total_steps,
        lr=learning_rate,
    )

    # Build animation frames
    frames = []
    for frame_idx in range(num_frames):
        s, b, mse = trajectory[frame_idx * subsample_stride]
        y_pred = _regression_predictions(x, s, b)

        frame = go.Frame(
            data=[
                go.Scatter(x=x, y=y, mode="markers", name="Data"),
                go.Scatter(x=x, y=y_pred, mode="lines", name="Regression Line"),
            ],
            name=f"Step {frame_idx}",
            layout=go.Layout(
                annotations=[
                    dict(
                        x=0.05,
                        y=1.1,
                        xref="paper",
                        yref="paper",
                        text=(
                            f"Schritt: {frame_idx}, "
                            f"a (Steigung): {s:.3f}, "
                            f"b (y-Achsenabschnitt): {b:.3f}, "
                            f"MSE: {mse:.3f}"
                        ),
                        showarrow=False,
                    )
                ]
            ),
        )
        frames.append(frame)

    # Initial figure
    initial_s, initial_b, _ = trajectory[0]
    fig = go.Figure(
        data=[
            go.Scatter(x=x, y=y, mode="markers", name="Data"),
            go.Scatter(x=x, y=_regression_predictions(x, initial_s, initial_b), mode="lines", name="Regression Line"),
        ],
        layout=go.Layout(
            xaxis=dict(title="Land Surface Temperature"),
            yaxis=dict(title="Active Fires"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                        ),
                    ],
                )
            ],
        ),
        frames=frames,
    )

    # Axis styling (kept behavior)
    fig.update_yaxes(range=(0, 1.1 * float(np.max(y))))
    fig.update_xaxes(zeroline=True, zerolinecolor="grey", zerolinewidth=2, showgrid=False)
    fig.update_yaxes(zeroline=True, zerolinecolor="grey", zerolinewidth=2, showgrid=False)

    fig.show()
