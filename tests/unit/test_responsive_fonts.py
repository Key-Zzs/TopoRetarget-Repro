from __future__ import annotations


def test_responsive_font_scaling_grows_with_figure():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from toporetarget.viz.responsive_fonts import install_responsive_font_scaling

    figure = plt.figure(figsize=(4, 3))
    axis = figure.add_subplot(111, projection="3d")
    axis.set_title("title")
    axis.set_xlabel("x")
    axis.text(0.0, 0.0, 0.0, "label")
    figure.text(0.1, 0.1, "figure text")
    figure.canvas.draw()
    connection, apply = install_responsive_font_scaling(figure)
    before = [text.get_fontsize() for text in figure.findobj(match=Text)]

    figure.set_size_inches(8, 6)
    figure.canvas.draw()
    apply()
    after = [text.get_fontsize() for text in figure.findobj(match=Text)]

    assert len(before) == len(after)
    assert all(new > old for old, new in zip(before, after, strict=True))
    assert max(after) <= 2.5 * max(before) + 1e-9
    figure.canvas.mpl_disconnect(connection)
    plt.close(figure)
