# hebbwatch

**Watch synaptic weights change in real time as a neural network learns.**

`hebbwatch` is a terminal-native observability tool for Hebbian learning dynamics. It renders a live heatmap of synaptic weights while a small neural system learns from synthetic input streams. The goal is simple: make learning visible while it is happening, not after training finishes.

```bash
pip install -e .
hebbwatch run --neurons 64 --inputs 8 --duration 30
```

## Why this exists

Most learning simulations run first and get analyzed later. You usually inspect a CSV, log, or static plot after the process is complete. That makes it hard to see structural learning dynamics as they unfold.

`hebbwatch` focuses on live observability:

- weight strengthening and decay
- sparsity emergence
- synaptic competition
- dominant pathway formation
- dead or inactive neurons
- instability and saturation

## Install

From the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e ".[dev]"
```

## Quick start

```bash
hebbwatch run
```

Try different dynamics:

```bash
hebbwatch run --rule hebbian --input-mode random
hebbwatch run --rule oja --input-mode oscillatory
hebbwatch run --rule competitive --input-mode correlated
hebbwatch run --neurons 96 --inputs 16 --duration 20 --learning-rate 0.02
```

Save and replay a session:

```bash
hebbwatch run --duration 8 --record recordings/my_run.hw
hebbwatch replay recordings/my_run.hw
```

Run a preset:

```bash
hebbwatch preset hebbwatch/presets/competitive.yaml
```

## File structure

```text
hebbwatch/
├── hebbwatch/
│   ├── cli.py
│   ├── config.py
│   ├── core/
│   ├── inputs/
│   ├── render/
│   ├── observability/
│   ├── replay/
│   ├── integrations/
│   └── presets/
├── examples/
├── tests/
├── scripts/
├── recordings/
├── assets/
└── docs/
```

## Architecture

`hebbwatch` separates the project into three main layers:

1. **Simulation** — generates neural activity and updates weights.
2. **Observability** — computes learning dynamics such as sparsity, entropy, competition, and collapse.
3. **Rendering** — displays the evolving weight matrix live in the terminal.

This separation keeps the project from becoming a one-file animation. The simulator can run without the renderer, the renderer can display saved replays, and future integrations can stream real model weights into the same observability layer.

## Learning rules

Supported rules:

- `hebbian` — classic co-activation strengthening with decay
- `oja` — normalized Hebbian learning for stable weights
- `competitive` — winner-biased Hebbian learning that forms clearer assemblies

Supported input streams:

- `random`
- `sparse`
- `correlated`
- `oscillatory`
- `drifting`

No public dataset is needed. The default demo uses synthetic correlated inputs so structure emerges immediately.

## Python API

```python
from hebbwatch.core.simulation import SimulationConfig, SimulationRunner

config = SimulationConfig(neurons=64, inputs=8, steps=240)
runner = SimulationRunner(config)

for state in runner.states():
    print(state.step, state.weights.mean())
```

## Tests

```bash
pytest
```

## Development commands

```bash
python scripts/benchmark_rendering.py
python scripts/generate_gif.py
python scripts/profile_simulation.py
```

## Roadmap

- smoother terminal color rendering
- GIF export
- live keyboard controls
- PyTorch layer heatmaps
- layer-wise backprop observability
- anomaly alerts for saturation/collapse
- side-by-side run comparison

## License

MIT
