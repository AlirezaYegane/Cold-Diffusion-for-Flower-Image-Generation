from __future__ import annotations

from pathlib import Path

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms


def build_image_transform(image_size: int = 64, train: bool = True):
    tfms = [
        transforms.Resize(72),
        transforms.CenterCrop(image_size),
    ]
    if train:
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(tfms)


class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 64,
        train_augment: bool = True,
        max_items: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "data" / "raw"
        self.image_dir = self.raw_dir / "jpg"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        setid = loadmat(self.raw_dir / "setid.mat")
        labels_mat = loadmat(self.raw_dir / "imagelabels.mat")
        labels = labels_mat["labels"].squeeze()

        split_map = {
            "train": setid["trnid"].squeeze(),
            "val": setid["valid"].squeeze(),
            "test": setid["tstid"].squeeze(),
        }
        if split not in split_map:
            raise ValueError(f"Unknown split: {split}")

        self.image_ids = split_map[split].tolist()
        if max_items is not None:
            self.image_ids = self.image_ids[:max_items]

        self.labels = labels
        self.transform = build_image_transform(
            image_size=image_size,
            train=(split == "train" and train_augment),
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int):
        image_id = int(self.image_ids[index])
        image_path = self.image_dir / f"image_{image_id:05d}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        image = Image.open(image_path).convert("RGB")
        x = self.transform(image)

        # labels are 1-based in the .mat file
        label = int(self.labels[image_id - 1])

        return {
            "image": x,
            "label": label,
            "image_id": image_id,
            "path": str(image_path),
        }