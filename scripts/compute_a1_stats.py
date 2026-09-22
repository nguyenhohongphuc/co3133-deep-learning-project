import json

from torchvision.datasets import FashionMNIST


with open("results/a1/splits/split_seed42.json", encoding="utf-8") as file:
    split = json.load(file)

dataset = FashionMNIST(
    root="data/",
    train=True,
    download=False,
)

# Raw pixels: uint8 in [0, 255].
# Convert to float in [0, 1], using only the fixed 54k train indices.
train_images = dataset.data[split["train_idx"]].float() / 255.0

mean = train_images.mean()
std = train_images.std(unbiased=False)

print(f"Train split size: {len(train_images)}")
print(f"Mean: {mean.item():.6f}")
print(f"Std: {std.item():.6f}")