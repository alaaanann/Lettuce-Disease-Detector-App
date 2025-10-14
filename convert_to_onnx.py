import torch
from torchvision import models
import torch.nn as nn
import os

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
# This MUST be the exact same architecture that you trained.

print("Rebuilding the MobileNetV3 model structure...")
# Define the number of classes your model was trained on
num_classes = 8 

# Load the MobileNetV3 model structure (without pre-trained weights)
model = models.mobilenet_v3_small(weights=None) 
last_layer_in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features=last_layer_in_features, out_features=num_classes)
print("Model structure is ready.")

# --- 2. LOAD YOUR TRAINED WEIGHTS ---
# The script will look for your .pth file in the same folder.
pth_file = 'lettuce_detector.pth'
if not os.path.exists(pth_file):
    print(f"Error: Trained model file '{pth_file}' not found.")
    print("Please make sure your trained model is in the same folder as this script.")
    exit()

# --- DEBUG: Print the shape of the model's final layer before loading weights ---
print(f"The rebuilt model's classifier expects {num_classes} classes.")
print(f"Shape of the new classifier's weight tensor: {model.classifier[-1].weight.shape}")

print(f"Loading trained weights from '{pth_file}'...")
# We use map_location=torch.device('cpu') to ensure the model can be loaded on any machine.
# UPDATED: Added weights_only=True to address the FutureWarning and for security.
model.load_state_dict(torch.load(pth_file, map_location=torch.device('cpu'), weights_only=True))
print("Weights loaded successfully.")

# Set the model to evaluation mode (this is important for conversion)
model.eval()

# --- 3. CONVERT THE MODEL TO ONNX ---
# Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
# This is needed by the ONNX exporter to trace the model's architecture.
dummy_input = torch.randn(1, 3, 224, 224) 
onnx_output_path = "lettuce_detector.onnx"

print(f"Exporting model to ONNX format at '{onnx_output_path}'...")
torch.onnx.export(model,
                  dummy_input,
                  onnx_output_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],   # The name for the input tensor
                  output_names=['output'], # The name for the output tensor
                  dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

print("\n--- Conversion Complete! ---")
print(f"Your model has been saved as '{onnx_output_path}'.")
print("You can now use this file with the HTML interface or on your Raspberry Pi.")

