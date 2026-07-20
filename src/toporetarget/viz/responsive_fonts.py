"""Responsive Matplotlib font scaling for interactive viewers."""

from __future__ import annotations

from typing import Any

_BASE_SIZE_ATTR = "_toporetarget_base_fontsize"


def _scale_for_figure(
    figure: Any,
    *,
    reference_pixels: tuple[float, float],
    minimum: float = 0.65,
    maximum: float = 2.5,
) -> float:
    width, height = (float(value) for value in figure.canvas.get_width_height())
    reference_width, reference_height = reference_pixels
    area_scale = ((width / reference_width) * (height / reference_height)) ** 0.5
    return max(minimum, min(maximum, area_scale))


def apply_responsive_font_scale(
    figure: Any,
    *,
    reference_pixels: tuple[float, float] | None = None,
    minimum: float = 0.65,
    maximum: float = 2.5,
) -> float:
    """Scale every Matplotlib Text artist according to the current window size.

    The unscaled size is stored per artist, so repeated resize events do not
    compound the scale.  ``figure.findobj`` also includes axis labels, tick
    labels, legends, annotations, and widget labels.
    """

    if reference_pixels is None:
        reference_pixels = (
            float(figure.get_size_inches()[0] * figure.dpi),
            float(figure.get_size_inches()[1] * figure.dpi),
        )
    scale = _scale_for_figure(
        figure,
        reference_pixels=reference_pixels,
        minimum=minimum,
        maximum=maximum,
    )
    from matplotlib.text import Text

    for text in figure.findobj(match=Text):
        base_size = getattr(text, _BASE_SIZE_ATTR, None)
        if base_size is None:
            base_size = float(text.get_fontsize())
            setattr(text, _BASE_SIZE_ATTR, base_size)
        text.set_fontsize(float(base_size) * scale)
    return scale


def install_responsive_font_scaling(
    figure: Any,
    *,
    minimum: float = 0.65,
    maximum: float = 2.5,
) -> tuple[int, Any]:
    """Install a resize callback and return ``(connection_id, apply)``.

    Viewers that recreate axes on frame changes should call the returned
    ``apply`` callback after each redraw so newly-created labels receive the
    current scale.
    """

    reference_pixels = (
        float(figure.get_size_inches()[0] * figure.dpi),
        float(figure.get_size_inches()[1] * figure.dpi),
    )

    def apply() -> float:
        return apply_responsive_font_scale(
            figure,
            reference_pixels=reference_pixels,
            minimum=minimum,
            maximum=maximum,
        )

    def on_resize(_event: Any) -> None:
        apply()
        figure.canvas.draw_idle()

    connection = figure.canvas.mpl_connect("resize_event", on_resize)
    apply()
    return connection, apply


__all__ = ["apply_responsive_font_scale", "install_responsive_font_scaling"]
