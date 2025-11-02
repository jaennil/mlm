from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from loguru import logger

from .config import (
    RAW_DATA_DIR,
    CLASS_FOLDERS, FOLDER_TO_ID,
    TRAIN_CSV, VAL_CSV, SEED
)

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.3),
    A.RandomFog(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class BeverageDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root = RAW_DATA_DIR

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row['class_folder'] / row['filename']).convert("RGB")
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, row['label']

def create_splits():
    data = []
    for folder in CLASS_FOLDERS:
        folder_path = RAW_DATA_DIR / folder
        if not folder_path.exists():
            logger.error(f"folder {folder_path} does not exist")
            continue
        for img in folder_path.glob("*.*"):
            if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                data.append({
                    "filename": img.name,
                    "class_folder": folder,
                    "label": FOLDER_TO_ID[folder]
                })
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    logger.info(f"данные разделены на train ({len(train_df)}), val ({len(val_df)})")
