"""
Gradio application for beverage package classification.

This application uses an ONNX model to classify images of beverage packages
into three categories: cans, paper cups, and plastic bottles.
"""

import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

# Configuration
APP_DIR = Path(__file__).parent
ONNX_PATH = APP_DIR / "model.onnx"
CLASS_NAMES = ["Жестяная банка", "Картонный стакан", "Пластиковая бутылка"]

# Image preprocessing (same as validation transform)
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Load ONNX model
if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found at {ONNX_PATH}. "
        "Please train the model first using: python train.py --config best"
    )

session = ort.InferenceSession(str(ONNX_PATH))


def predict(img: Image.Image):
    """
    Predict the class of a beverage package image.
    
    Args:
        img: PIL Image
        
    Returns:
        str: Prediction result with class name and confidence score
    """
    
    try:
        # Preprocess image
        x = transform(image=np.array(img))["image"].numpy()
        x = np.expand_dims(x, 0)
        
        # Run inference
        pred = session.run(None, {"input": x})[0]
        
        # Get prediction
        idx = np.argmax(pred)
        confidence = pred[0][idx]
        
        # Format result
        result = f"**{CLASS_NAMES[idx]}**\n\nУверенность: {confidence:.2%}"
        
        # Add probabilities for all classes
        result += "\n\n### Вероятности всех классов:"
        for _, (class_name, prob) in enumerate(zip(CLASS_NAMES, pred[0])):
            result += f"\n- {class_name}: {prob:.2%}"
        
        return result
        
    except Exception as e:
        return f"Ошибка при обработке изображения: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение упаковки напитка"),
    outputs=gr.Markdown(label="Результат классификации"),
    title="🥫 Классификатор упаковок напитков",
    description="""
    Загрузите изображение упаковки напитка, и модель определит её тип:
    - 🥫 **Жестяная банка** (металлические банки для напитков)
    - ☕ **Картонный стакан** (бумажные стаканчики для кофе/чая)
    - 🍼 **Пластиковая бутылка** (пластиковые бутылки для воды/соков)
    """,
    examples=[
        # Add paths to example images if available
    ],
    article="""
    ### О проекте
    
    Модель обучена с использованием transfer learning на архитектуре ConvNeXt Tiny.
    Достигнутая точность на валидационной выборке: **100%**.
    
    **Технологии:**
    - PyTorch + timm для обучения
    - ONNX для оптимизированного инференса
    - Gradio для веб-интерфейса
    
    **Примечание:** Лучше всего работает с четкими изображениями упаковок на однородном фоне.
    """,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
