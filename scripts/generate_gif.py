import os
import numpy as np
from PIL import Image

os.makedirs("assets", exist_ok=True)

frames = []
size = 64

weights = np.random.normal(0, 0.2, (size, size))

for t in range(160):
    activity = np.random.normal(0, 1, size)
    activity = np.tanh(activity)

    weights += 0.015 * np.outer(activity, activity)
    weights *= 0.995

    x = weights.copy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    img = (x * 255).astype(np.uint8)

    frame = Image.fromarray(img).resize((512, 512))
    frames.append(frame)

frames[0].save(
    "assets/demo.gif",
    save_all=True,
    append_images=frames[1:],
    duration=50,
    loop=0,
)

print("Saved GIF to assets/demo.gif")