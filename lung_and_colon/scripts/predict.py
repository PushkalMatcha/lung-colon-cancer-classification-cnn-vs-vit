import torch
from torchvision import transforms
from PIL import Image
import argparse
import json

from model import get_model
from dataset import CLASS_NAMES

# --- Configuration ---
MODEL_PATH = 'results/saved_models/best_model.pth'
IMAGE_SIZE = 224

def predict_single_image(image_path, model_path):
    """
    Predicts the class for a single image.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the saved model state dictionary.

    Returns:
        tuple: A tuple containing the predicted class name and the confidence score.
    """
    # 1. Setup: Device, Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    
    # Load the model
    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Image Preprocessing
    # Define the same transformations as used for validation/testing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None
        
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    # 3. Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Predict the class of a single histopathology image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the trained model file.')
    
    args = parser.parse_args()

    # Perform prediction
    predicted_class, confidence = predict_single_image(args.image_path, args.model_path)
    
    if predicted_class:
        print("\n--- Prediction Result ---")
        print(f"🖼️ Image Path: {args.image_path}")
        print(f"🩺 Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print("-------------------------\n")