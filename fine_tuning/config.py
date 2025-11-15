from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from loguru import logger
import random
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

APP_DIR = PROJ_ROOT / "app"
APP_DIR.mkdir(exist_ok=True)
ONNX_PATH = APP_DIR / "model.onnx"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"

MODELS_DIR = PROJ_ROOT / "models"
BEST_MODEL_PTH = MODELS_DIR / "best_model.pth"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX = FIGURES_DIR / "confusion_matrix.png"
LEARNING_CURVES = FIGURES_DIR / "learning_curves.png"

CLASS_FOLDERS = ["can", "paper_cup", "plastic_bottle"]
CLASS_NAMES = ["Жестяная банка", "Картонный стакан", "Пластиковая бутылка"]
NUM_CLASSES = len(CLASS_NAMES)

FOLDER_TO_ID: Dict[str, int] = {folder: idx for idx, folder in enumerate(CLASS_FOLDERS)}
SEED = 42

@dataclass
class TrainingConfig:
    model_name: str = "resnet34"
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 10
    freeze_backbone: bool = True
    unfreeze_epoch: int = 5
    seed: int = 42
    optimizer: str = "Adam"
    scheduler: str = ""
    scheduler_step: int = 5
    scheduler_gamma: float = 0.1

@dataclass
class ResNet34Config(TrainingConfig):
    model_name: str = "resnet34"
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 12
    unfreeze_epoch: int = 6
    scheduler: str = ""

@dataclass
class ConvNextConfig(TrainingConfig):
    model_name: str = "convnext_tiny"
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 15
    unfreeze_epoch: int = 7
    scheduler: str = "StepLR"
    scheduler_step: int = 5
    scheduler_gamma: float = 0.1

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"seed зафиксирован: {seed}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
