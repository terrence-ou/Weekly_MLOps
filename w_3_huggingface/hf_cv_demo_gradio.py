import requests
import gradio as gr
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image


def predict_from_url(url):
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    result = f"Predicted class: {model.config.id2label[predicted_class_idx]}"
    return result

if __name__ == "__main__":
    iface = gr.Interface(fn=predict_from_url, inputs="text", outputs="text")
    iface.launch()