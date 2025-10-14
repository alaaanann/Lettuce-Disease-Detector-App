import torch
from torchvision import models, transforms
import torch.nn as nn
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io
import os

# --- 1. INITIALIZE THE FLASK APP ---
app = Flask(__name__)

# --- 2. DEFINE MODEL PARAMETERS AND LOAD THE MODEL ---
# This list is now hardcoded. It MUST match the order from your training.
CLASS_NAMES = [
    'Bacterial',
    'Downy_mildew_on_lettuce',
    'Healthy',
    'Powdery_mildew_on_lettuce',
    'Septoria_blight_on_lettuce',
    'Shepherd_purse_weeds',
    'Viral',
    'Wilt_and_leaf_blight_on_lettuce',
]
num_classes = len(CLASS_NAMES)
print(f"Model configured for {num_classes} classes: {CLASS_NAMES}")

try:
    # Rebuild the model architecture
    model = models.mobilenet_v3_small(weights=None)
    last_layer_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=last_layer_in_features, out_features=num_classes)
    
    # Load your saved weights
    model.load_state_dict(torch.load('lettuce_detector.pth', map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    print("PyTorch model loaded successfully and ready for predictions.")

except Exception as e:
    print(f"--- FATAL ERROR: Could not load the model. ---")
    print(f"Error details: {e}")
    print("Please ensure 'lettuce_detector.pth' is in the same folder as this script.")
    model = None


# --- 3. DEFINE THE IMAGE TRANSFORMATION PIPELINE ---
# This is the exact same pipeline from your proven test script.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# --- 4. DEFINE THE WEB ROUTES ---

# This route serves the main HTML page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# This route handles the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read the image file from the request
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Apply the transformations to the image
        input_tensor = transform(image).unsqueeze(0) # Add a batch dimension
        
        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
        
        # Get the human-readable class name
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        
        # Return the result as JSON
        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image.'}), 500


# --- 5. RUN THE FLASK APP ---
if __name__ == '__main__':
    # Use port 5001 to avoid conflicts with other common applications
    app.run(debug=True, port=5001)

