import numpy as np

from hebbwatch.render.heatmap import render_heatmap


def test_heatmap_outputs_text():
    text = render_heatmap(np.random.randn(8, 4))
    assert len(text.plain) > 0
