import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import pandas as pd

csv_path=r"C:\Users\Tanee\OneDrive\Desktop\C anew\LUC\fertilizer_dataset_Brief.csv"

# Label mappings for classes
class_mappings = {
    0: "bacterial_leaf_blight",
    1: "healthy",
    2: "leaf_scald",
    3: "tungro"
}

# Read the fertilizer data from a CSV file
def load_fertilizer_data(csv_path):
    
    df = pd.read_csv(csv_path)
    fertilizer_data = {}
    for index, row in df.iterrows():
        disease_name = row['Disease']
        fertilizer_data[disease_name] =       {
            "Nitrogen Fertilizer": f"{row['Nitrogen Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Phosphorus Fertilizer": f"{row['Phosphorus Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Potassium Fertilizer": f"{row['Potassium Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}",
            "Zinc Fertilizer": f"{row['Zinc Fertilizer']} {row['Fertilizer Quantity(kg/acre)']}"
        }
    return fertilizer_data

fertilizer_recommendations = load_fertilizer_data(csv_path)

# Load model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model_path = r"C:\Users\Tanee\OneDrive\Desktop\C anew\LUC\supervised_resnet50_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Predict image label
def predict_image(image):
    if image is None:
        return "Error: No image provided"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            predicted_label = class_mappings[predicted_class]
    except Exception as e:
        return f"Error during prediction: {str(e)}"
    return predicted_label

# Fertilizer suggestions
def get_fertilizer_suggestions(disease_label):
    fertilizer_info = fertilizer_recommendations.get(disease_label, None)
    if fertilizer_info:
        recommendations = "\n".join([f"{key}: {value}" for key, value in fertilizer_info.items()])
    else:
        recommendations = "No fertilizer recommendations available."
    return recommendations

# Gradio Interface
def process_image(image):
    return predict_image(image)

def suggest_fertilizer(disease_label):
    return get_fertilizer_suggestions(disease_label)

# Create Gradio interface with inline styling via CSS
with gr.Blocks(css="""
    #image_input { border: 3px solid #4CAF50; background-color: #E8F5E9; padding: 10px; }
    #disease_label_output { color: #1A237E; font-weight: bold; background-color: #E3F2FD; }
    #fertilizer_button { background-color: #FDD835; color: #212121; font-weight: bold; }
    #fertilizer_output { background-color: #FFF9C4; color: #BF360C; padding: 10px; }
""") as iface:
    image_input = gr.Image(type="pil", label="Upload Rice Leaf Image", elem_id="image_input")
    disease_label_output = gr.Textbox(label="Predicted Disease Label", elem_id="disease_label_output")
    fertilizer_button = gr.Button("Suggest Fertilizer", elem_id="fertilizer_button")
    fertilizer_output = gr.Textbox(label="Suggested Fertilizer Recommendations", elem_id="fertilizer_output")
    
    image_input.change(process_image, inputs=image_input, outputs=disease_label_output)
    fertilizer_button.click(suggest_fertilizer, inputs=disease_label_output, outputs=fertilizer_output)

iface.launch()
