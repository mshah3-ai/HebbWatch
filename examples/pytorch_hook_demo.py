try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Install PyTorch first: pip install 'hebbwatch[torch]'")

from hebbwatch.integrations.pytorch import watch_linear_layers

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

def callback(name, weight):
    print(name, weight.shape, abs(weight).mean())

handle = watch_linear_layers(model, callback)
model(torch.randn(2, 8))
handle.remove()
