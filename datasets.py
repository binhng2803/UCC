import os
import numpy as np
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader


class SegDataset(Dataset):
    def __init__(self, root_dir="data", phase="warmup", split="train", transform=None):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / phase / "img" / split
        self.ann_dir = self.root_dir / phase / "ann" / split
        self.transform = transform
        self.img_list = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / self.img_list[idx])
        br_enhancer = ImageEnhance.Brightness(img)
        br = 0.75
        img = br_enhancer.enhance(br)
        ct_enhancer = ImageEnhance.Contrast(img)
        ct = 2
        img = ct_enhancer.enhance(ct)
        img = np.array(img)
        
        ann_path = self.ann_dir / f"{Path(self.img_list[idx]).stem}.png"
        ann = np.array(Image.open(ann_path))

        if self.transform:
            augmented = self.transform(image=img, mask=ann)
            img = augmented["image"]
            ann = augmented["mask"]

        return img, ann


class SegDataModule(pl.LightningDataModule):
    def __init__(self, root_dir="data", phase="warmup", batch_size: int = 8):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="train",
            transform=A.Compose([A.RandomCrop(380, 380),
                                A.RandomRotate90(p=0.5),
                                # A.RandomBrightnessContrast(brightness_limit=(0.5, 0.99), contrast_limit=(-0.35, -0.25), p=0.7),
                                # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                                # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                ToTensorV2(),]),
        )
        self.valid = SegDataset(
            root_dir=self.root_dir,
            phase=self.phase,
            split="valid",
            transform=A.Compose([#A.RandomBrightnessContrast(brightness_limit=(0.99, 0.99), contrast_limit=(-0.3, -0.3), p=1.),
                                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                ToTensorV2(),]),
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.valid, batch_size=1, shuffle=False, num_workers=1, drop_last=True)


if __name__ == "__main__":
    ds = SegDataset()
    img, ann = ds[10]
    print("Done!")
