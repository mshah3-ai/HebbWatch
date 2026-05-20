from __future__ import annotations

from hebbwatch.integrations.hooks import WatchHandle


def watch_linear_layers(model, callback):
    """Attach lightweight hooks to PyTorch Linear layers when torch is installed."""
    try:
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch integration requires installing hebbwatch[torch].") from exc

    handles = []

    def make_hook(name):
        def hook(module, _inputs, _output):
            weight = module.weight.detach().cpu().numpy()
            callback(name, weight)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name or "linear")))

    return WatchHandle(lambda: [h.remove() for h in handles])
