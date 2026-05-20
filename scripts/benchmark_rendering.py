from time import perf_counter

from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.render.heatmap import render_heatmap

runner = SimulationRunner(SimulationConfig(neurons=128, inputs=64, steps=200))
start = perf_counter()
for state in runner.states():
    render_heatmap(state.weights)
elapsed = perf_counter() - start
print(f"Rendered 200 frames in {elapsed:.3f}s ({200 / elapsed:.1f} fps equivalent)")
