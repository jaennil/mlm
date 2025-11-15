import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

APP_DIR = Path(__file__).parent
ONNX_PATH = APP_DIR / "model.onnx"
CLASS_NAMES = ["Жестяная банка", "Картонный стакан", "Пластиковая бутылка"]

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found at {ONNX_PATH}. "
        "Please train the model first"
    )

session = ort.InferenceSession(str(ONNX_PATH))


def predict(img: Image.Image):
    try:
        x = transform(image=np.array(img))["image"].numpy()
        x = np.expand_dims(x, 0)
        
        pred = session.run(None, {"input": x})[0]
        
        idx = np.argmax(pred)
        confidence = pred[0][idx]
        
        result = f"**{CLASS_NAMES[idx]}**\n\nУверенность: {confidence:.2%}"
        
        result += "\n\n### Вероятности всех классов:"
        for _, (class_name, prob) in enumerate(zip(CLASS_NAMES, pred[0])):
            result += f"\n- {class_name}: {prob:.2%}"
        
        return result
        
    except Exception as e:
        return f"Ошибка при обработке изображения: {str(e)}"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение упаковки напитка"),
    outputs=gr.Markdown(label="Результат классификации"),
    title="Классификатор упаковок напитков",
    description="""
    Загрузите изображение упаковки напитка, и модель определит её тип:
    - **Жестяная банка** (металлические банки для напитков)
    - **Картонный стакан** (бумажные стаканчики для кофе/чая)
    - **Пластиковая бутылка** (пластиковые бутылки для воды/соков)
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
