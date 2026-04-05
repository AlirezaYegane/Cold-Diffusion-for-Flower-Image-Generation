from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

img_dir = Path("data/raw/jpg")
image_paths = sorted(img_dir.glob("*.jpg"))[:9]

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, path in zip(axes.flatten(), image_paths):
    img = Image.open(path).convert("RGB")
    ax.imshow(img)
    ax.set_title(path.name, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()
