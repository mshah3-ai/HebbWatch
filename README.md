# hebbwatch

**Real-time observability for neural learning dynamics.**

`hebbwatch` is a terminal-native observability tool for Hebbian learning systems. It renders a live heatmap of synaptic weights while a neural network learns from streaming synthetic input activity, allowing you to watch structure, sparsity, competition, and stabilization emerge in real time.

Instead of inspecting static logs after training finishes, `hebbwatch` makes the learning process itself observable.

---

## Live Demo

![hebbwatch demo](assets/demo.gif)

```bash
pip install -e .
hebbwatch run --neurons 64 --inputs 8 --duration 30
```

---

## Why this exists

Modern ML tooling exposes:
- loss curves
- gradients
- scalar metrics
- GPU utilization

but the internal structural dynamics of learning remain mostly invisible during training.

As neural systems learn:
- some synapses strengthen aggressively
- others decay toward zero
- subnetworks compete for dominance
- inactive neurons collapse
- sparse assemblies emerge

These dynamics are usually inspected only after training through static plots or offline analysis.

`hebbwatch` makes these structural learning dynamics observable while they happen.

---

## Features

- live synaptic weight heatmaps
- real-time sparsity and competition metrics
- multiple Hebbian learning rules
- synthetic correlated input generators
- replayable learning sessions
- terminal-native rendering
- lightweight dependency stack
- extensible observability architecture
- future PyTorch observability hooks

---

## Install

From the project folder:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -e ".[dev]"
```

---

## Quick start

Run the default live simulation:

```bash
hebbwatch run
```

Try different dynamics:

```bash
hebbwatch run --rule hebbian --input-mode random
hebbwatch run --rule oja --input-mode oscillatory
hebbwatch run --rule competitive --input-mode correlated

hebbwatch run \
  --neurons 96 \
  --inputs 16 \
  --duration 20 \
  --learning-rate 0.02
```

---

## Replay system

Record a session:

```bash
hebbwatch run --duration 8 --record recordings/my_run.hw
```

Replay it later:

```bash
hebbwatch replay recordings/my_run.hw
```

This allows:
- rewindable learning sessions
- side-by-side comparisons
- GIF generation
- temporal observability analysis

---

## Presets

Run predefined configurations:

```bash
hebbwatch preset hebbwatch/presets/competitive.yaml
```

Available presets include:
- competitive assemblies
- sparse dynamics
- oscillatory systems
- chaotic attractor modes

---

## Terminal Observability

`hebbwatch` exposes:

- weight entropy
- sparsity emergence
- competition indices
- dead neuron detection
- dominant pathway formation
- stabilization dynamics
- synaptic collapse detection

directly inside the terminal while training occurs.

The goal is not post-hoc visualization.

The goal is live neural observability.

---

## Architecture

`hebbwatch` separates the project into three independent layers:

### 1. Simulation
Generates neural activity and updates synaptic weights using configurable learning rules.

### 2. Observability
Computes structural learning metrics such as:
- sparsity
- entropy
- competition
- collapse
- stabilization

### 3. Rendering
Displays evolving synaptic structure in real time through terminal-native heatmaps and metric panels.

This separation prevents the project from becoming a one-file animation demo:
- the simulator can run independently
- the renderer can replay recorded sessions
- future integrations can stream real model weights into the same observability layer

---

## Learning rules

Supported learning rules:

- `hebbian`
  - classic co-activation strengthening with decay

- `oja`
  - normalized Hebbian learning for stable bounded weights

- `competitive`
  - winner-biased Hebbian learning that forms stronger assemblies

---

## Input streams

Supported synthetic input generators:

- `random`
- `sparse`
- `correlated`
- `oscillatory`
- `drifting`

No public dataset is required.

The default demo uses correlated synthetic inputs so emergent structure appears immediately.

---

## Python API

```python
from hebbwatch.core.simulation import (
    SimulationConfig,
    SimulationRunner,
)

config = SimulationConfig(
    neurons=64,
    inputs=8,
    steps=240,
)

runner = SimulationRunner(config)

for state in runner.states():
    print(state.step, state.weights.mean())
```

---

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
│
├── examples/
├── tests/
├── scripts/
├── recordings/
├── assets/
└── docs/
```

---

## Development commands

Generate benchmark metrics:

```bash
python scripts/benchmark_rendering.py
```

Generate the README GIF:

```bash
python scripts/generate_gif.py
```

Profile simulation performance:

```bash
python scripts/profile_simulation.py
```

Run tests:

```bash
pytest
```

---

## Roadmap

- smoother terminal color rendering
- temporal rewind + replay scrubbing
- side-by-side run comparison
- PyTorch layer observability hooks
- transformer attention visualization
- layer-wise backprop observability
- anomaly alerts for saturation/collapse
- GPU-accelerated rendering
- WebSocket streaming dashboard

---

## Inspiration

`hebbwatch` is inspired by a simple idea:

> learning dynamics are computation, not just the final model.

Most neural systems evolve through:
- competition
- reinforcement
- stabilization
- collapse
- sparsification

but these dynamics are usually hidden behind logs and scalar metrics.

`hebbwatch` exposes those dynamics directly.

---

## License

MIT
