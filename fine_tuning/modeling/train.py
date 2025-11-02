import torch
import timm
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

from ..dataset import BeverageDataset, train_transform, val_transform, create_splits
from ..config import (
    TrainingConfig, ONNX_PATH,
    TRAIN_CSV, VAL_CSV, BEST_MODEL_PTH, CONFUSION_MATRIX, LEARNING_CURVES,
    CLASS_NAMES, seed_everything
)

def train_model(config: TrainingConfig):
    logger.info(f"starting training with config: {config}")
    seed_everything(config.seed)

    logger.info("creating train/validation splits...")
    create_splits()

    train_ds = BeverageDataset(TRAIN_CSV, train_transform)
    val_ds = BeverageDataset(VAL_CSV, val_transform)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    logger.info(f"train dataset size: {len(train_ds)}, val dataset size: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"using device: {device}")

    model = timm.create_model(config.model_name, pretrained=True, num_classes=len(CLASS_NAMES)).to(device)

    logger.info(f"model architecture: {config.model_name}")

    if config.freeze_backbone:
        logger.info("freezing backbone weights...")
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    best_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(config.epochs):
        if config.unfreeze_epoch and epoch == config.unfreeze_epoch:
            logger.info(f"unfreezing all weights at epoch {epoch}")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=config.lr / 10)

        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds.extend(model(xb).argmax(1).cpu().numpy())
                labels.extend(yb.cpu().numpy())

        acc = accuracy_score(labels, preds)
        val_accuracies.append(acc)
        logger.info(f"epoch {epoch}: loss = {avg_loss:.4f}, val_acc = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PTH)
            logger.info(f"new best model saved with accuracy: {best_acc:.4f}")

    logger.info("geterating confusion matrix...")
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"confusion matrix - {config.model_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX)
    plt.close()
    logger.success(f"confusion matrx saved to {CONFUSION_MATRIX}")

    logger.info("generating learning curves...")
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('training loss over epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(val_accuracies, label='validation accuracy', color='green')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_title('validation accuracy over epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(LEARNING_CURVES)
    plt.close()
    logger.success(f"learning curves saved to {LEARNING_CURVES}")

    logger.info("exporting model to ONNX...")
    model.eval()
    model = model.to('cpu')
    dummy = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy, ONNX_PATH, export_params=True, do_constant_folding=True,
                      input_names=["input"], output_names=["output"],
                      opset_version=18)
    logger.success(f"ONNX сохранён: {ONNX_PATH}")
    logger.success(f"training complete. best accuracy: {best_acc:.3f}")

    return {
        "best_accuracy": best_acc,
        "model_path": BEST_MODEL_PTH,
        "onnx_path": ONNX_PATH,
        "confusion_matrix_path": CONFUSION_MATRIX,
        "learning_curves_path": LEARNING_CURVES
    }
