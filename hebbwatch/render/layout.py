from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.panel import Panel

from hebbwatch.core.network import NetworkState
from hebbwatch.render.heatmap import render_heatmap
from hebbwatch.render.stats_panel import build_stats_table


def build_live_layout(state: NetworkState, rule: str, input_mode: str, width: int, height: int) -> Group:
    heatmap = render_heatmap(state.weights, width=width, height=height)
    title = "hebbwatch — live synaptic weight observability"
    heatmap_panel = Panel(Align.center(heatmap), title=title, border_style="cyan")
    stats_panel = Panel(build_stats_table(state, rule, input_mode), title="dynamics", border_style="green")
    return Group(heatmap_panel, stats_panel)
