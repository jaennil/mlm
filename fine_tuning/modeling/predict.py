import torch
from ..dataset import val_transform
from timm import create_model
from ..config import ConvNextConfig, BEST_MODEL_PTH, CLASS_NAMES

def predict_single(image_path: str, model_path: str = BEST_MODEL_PTH):
    config = ConvNextConfig()
    model = create_model(config.model_name, pretrained=False, num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    from PIL import Image
    import numpy as np
    image = np.array(Image.open(image_path).convert("RGB"))
    transformed = val_transform(image=image)["image"].unsqueeze(0)
    with torch.no_grad():
        pred = model(transformed).argmax(1).item()
    return CLASS_NAMES[pred]
