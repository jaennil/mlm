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


def train_model(config: TrainingConfig) -> dict:
    logger.info(f"Starting training with config: {config}")
    seed_everything(config.seed)

    logger.info("Creating train/validation splits...")
    create_splits()

    train_ds = BeverageDataset(TRAIN_CSV, train_transform)
    val_ds = BeverageDataset(VAL_CSV, val_transform)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size,
        num_workers=0
    )

    logger.info(f"Train dataset size: {len(train_ds)}, Val dataset size: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = timm.create_model(
        config.model_name, 
        pretrained=True, 
        num_classes=len(CLASS_NAMES)
    ).to(device)
    logger.info(f"Model architecture: {config.model_name}")

    if config.freeze_backbone:
        logger.info("Freezing backbone weights...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.lr
    )

    scheduler = None
    if config.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.scheduler_step, 
            gamma=config.scheduler_gamma
        )
        logger.info(
            f"Using StepLR scheduler: "
            f"step={config.scheduler_step}, gamma={config.scheduler_gamma}"
        )

    best_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(config.epochs):
        if config.unfreeze_epoch and epoch == config.unfreeze_epoch:
            logger.info(f"Unfreezing all weights at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config.lr / 10
            )
            
            if scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=config.scheduler_step,
                    gamma=config.scheduler_gamma
                )

        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        final_loss = avg_loss

        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if scheduler:
            old_lr = current_lr
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                logger.info(f"Learning rate updated: {old_lr:.2e} -> {new_lr:.2e}")
        
        logger.info(
            f"Epoch {epoch}: "
            f"loss={avg_loss:.4f}, "
            f"val_acc={val_acc:.4f}, "
            f"lr={current_lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PTH)
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES, 
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {config.model_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX, dpi=150)
    plt.close()
    logger.success(f"Confusion matrix saved to {CONFUSION_MATRIX}")

    logger.info("Generating learning curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accuracies, label='Validation Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(LEARNING_CURVES, dpi=150)
    plt.close()
    logger.success(f"Learning curves saved to {LEARNING_CURVES}")

    logger.info("Exporting model to ONNX...")
    model.eval()
    model = model.to('cpu')
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input, 
        ONNX_PATH, 
        export_params=True, 
        do_constant_folding=True,
        input_names=["input"], 
        output_names=["output"],
        opset_version=18
    )
    logger.success(f"ONNX model saved: {ONNX_PATH}")
    logger.success(f"Training complete. Best accuracy: {best_acc:.3f}")

    return {
        "best_accuracy": best_acc,
        "final_loss": final_loss,
        "model_path": BEST_MODEL_PTH,
        "onnx_path": ONNX_PATH,
        "confusion_matrix_path": CONFUSION_MATRIX,
        "learning_curves_path": LEARNING_CURVES
    }
