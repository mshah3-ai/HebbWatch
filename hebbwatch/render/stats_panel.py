from __future__ import annotations

from rich.table import Table

from hebbwatch.core.metrics import competition_index, dead_neurons, dominant_connections, sparsity, weight_entropy
from hebbwatch.core.network import NetworkState


def build_stats_table(state: NetworkState, rule: str, input_mode: str) -> Table:
    weights = state.weights
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Step", f"{state.step:,}")
    table.add_row("Rule", rule)
    table.add_row("Input", input_mode)
    table.add_row("Mean |w|", f"{abs(weights).mean():.4f}")
    table.add_row("Max |w|", f"{abs(weights).max():.4f}")
    table.add_row("Sparsity", f"{100 * sparsity(weights):.1f}%")
    table.add_row("Entropy", f"{weight_entropy(weights):.2f}")
    table.add_row("Competition", f"{competition_index(weights):.3f}")
    table.add_row("Dead neurons", str(dead_neurons(weights)))
    table.add_row("Top links", str(dominant_connections(weights)))
    return table
