# Architecture

hebbwatch has three layers: simulation, observability, and rendering.

- `core/` owns the neural update loop.
- `observability/` measures dynamics from weight matrices.
- `render/` turns those matrices and metrics into live terminal output.

The design allows future backends, including PyTorch hooks, without rewriting the terminal interface.
