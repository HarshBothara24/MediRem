import torch
from flask import Flask, redirect, request, render_template, jsonify, url_for
from PIL import Image
import torchvision.transforms as transforms
import io
import re
from paddleocr import PaddleOCR
import numpy as np
import requests
import os
import tempfile

from model import MedReminderModel  # Assuming OCRModel is defined elsewhere, or you can define it

# Initialize Flask app
app = Flask(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your pre-trained model
num_classes = 10  # Adjust this number based on the number of categories or outputs your model predicts
model = MedReminderModel(num_classes=num_classes)  # Pass num_classes when initializing the model
model.load_state_dict(torch.load("med_reminder.pth"))
model = model.to(device)
model.eval()

# Initialize PaddleOCR
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

# Define the image transformation (same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Regex pattern for medicine and syrup details
medicine_pattern = re.compile(r"([a-zA-Z\s]+(?:tab|cap|tablet|capsule))\s*(\d+mg)\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")
syrup_pattern = re.compile(r"([a-zA-Z\s]+(?:syrup))\s*(\d+x|\d+\s*times)\s*(\d+\s*days|\d+\s*week)")

# Process model output (adjust this function as per your model's output)
def process_model_output(output):
    # This function should convert the model's raw output into structured data.
    # For example, you could use PaddleOCR for OCR and regex to find medicine details.
    
    # Extract text from the output image using PaddleOCR
    ocr_results = ocr_engine.ocr(output, cls=True)
    text = "\n".join([line[1][0] for line in ocr_results[0]])
    
    # Extract medicines and syrups from the text
    medicines = []
    syrups = []
    
    # Use regex to match medicine patterns
    for match in medicine_pattern.finditer(text):
        medicine_details = {
            "name": match.group(1),
            "dosage": match.group(2),
            "frequency": match.group(3),
            "duration": match.group(4)
        }
        medicines.append(medicine_details)
    
    # Use regex to match syrup patterns
    for match in syrup_pattern.finditer(text):
        syrup_details = {
            "name": match.group(1),
            "frequency": match.group(2),
            "duration": match.group(3)
        }
        syrups.append(syrup_details)

    return {"medicines": medicines, "syrups": syrups}

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/apidata')
def apidata():
    # return jsonify(process_model_output)
    return "Testing Successful 200"

# Upload image and run inference
@app.route('/predict', methods=['POST'])
def predict():
    image = None  # Placeholder for the image data

    try:
        # Check if a file is uploaded
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")

        # Check if an image URL is provided
        elif 'image_url' in request.form or 'image_url' in request.json:
            image_url = request.form.get('image_url') if 'image_url' in request.form else request.json.get('image_url')
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Check if the content type is image
            if 'image' not in response.headers['Content-Type']:
                return jsonify({"error": "The provided URL does not point to an image"})

            # Open the image from the response
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

        else:
            return jsonify({"error": "No file or image URL provided"})

        # Convert image to NumPy array
        image_np = np.array(image)

        # Extract text using PaddleOCR
        ocr_results = ocr_engine.ocr(image_np, cls=True)
        extracted_text = "\n".join([line[1][0] for line in ocr_results[0]])

        # Process the extracted text for medicines and syrups
        extracted_medicine_data = extract_medicine_data(extracted_text)

        # Return results as JSON
        return jsonify({
            "extracted_text": extracted_text,
            "extracted_medicine_data": extracted_medicine_data
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching image from URL: {str(e)}"})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})
        
def extract_medicine_data(text):
    # You can customize this function to process the extracted text further
    # Here, I am assuming the text includes medicine and dosage details
    medicine_data = []

    lines = text.split('\n')
    for line in lines:
        if 'Tab' in line or 'TAB' in line or 'tab' in line or 'Cap' in line or 'CAP' in line or 'cap' in line or 'Syrup' in line or 'SYRUP' in line or 'syrup' in line or 'SYP' in line or 'syp' in line or 'Syp' in line or 'days' in line or 'Days' in line or 'times a day' in line or 'Times a day' in line:
            medicine_data.append(line.strip())

    return medicine_data

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
