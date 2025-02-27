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
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

from model import MedReminderModel  # Assuming OCRModel is defined elsewhere, or you can define it

# Initialize Flask app
app = Flask(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()
scheduler.start()


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

# Function to schedule reminders based on extracted medicine data
def schedule_reminders(medicine_data):
    for medicine in medicine_data:
        name = medicine.get("name")
        frequency = medicine.get("frequency", "").lower()
        duration = medicine.get("duration", "").lower()

        # Determine the number of times per day
        times_per_day = 1
        if "2x" in frequency or "2 times" in frequency:
            times_per_day = 2
        elif "3x" in frequency or "3 times" in frequency:
            times_per_day = 3
        elif "4x" in frequency or "4 times" in frequency:
            times_per_day = 4

        # Set start time (current time) and interval
        start_time = datetime.now()
        interval_hours = 24 // times_per_day

        # Determine duration in days
        duration_days = 1
        if "week" in duration:
            duration_days = int(duration.split()[0]) * 7
        elif "day" in duration:
            duration_days = int(duration.split()[0])

        end_time = start_time + timedelta(days=duration_days)

        # Schedule reminders
        for i in range(times_per_day):
            reminder_time = start_time + timedelta(hours=i * interval_hours)
            if reminder_time < end_time:
                scheduler.add_job(
                    func=lambda: print(f"Reminder: Take {name} at {reminder_time.strftime('%H:%M')}"),
                    trigger="date",
                    run_date=reminder_time
                )

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
    try:
        image = None  # Placeholder for the image data

        # Check if an image file is uploaded
        if 'file' in request.files and request.files['file']:
            uploaded_file = request.files['file']
            image = Image.open(uploaded_file).convert("RGB")

        # Check if an image URL is provided
        if image is None:
            image_url = request.form.get('image_url') or (request.get_json(silent=True) or {}).get('image_url')

            if image_url:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP issues

                # Validate that the response contains an image
                if 'image' not in response.headers.get('Content-Type', ''):
                    return jsonify({"error": "The provided URL does not point to an image"})

                # Load the image from the response
                image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Ensure an image was successfully loaded
        if image is None:
            return jsonify({"error": "No valid image file or URL provided."})

        # Convert image to NumPy array
        image_np = np.array(image)

        # Debugging: Print image shape
        print("Image shape:", image_np.shape)

        # Perform OCR using PaddleOCR
        ocr_results = ocr_engine.ocr(image_np, cls=True)

        # Debugging: Print raw OCR results
        print("OCR Raw Results:", ocr_results)

        # Ensure OCR results are valid
        if not ocr_results or not ocr_results[0]:
            return jsonify({"error": "No text detected in the image."})

        # Extract text from OCR results
        extracted_text = "\n".join([line[1][0] for line in ocr_results[0]])

        # Debugging: Print extracted text
        print("Extracted Text:", extracted_text)

        # Process extracted text for medicine-related data
        extracted_medicine_data = extract_medicine_data(extracted_text)

        return jsonify({
            "extracted_text": extracted_text,
            "extracted_medicine_data": extracted_medicine_data,
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


# if __name__ == "__main__":
#     app.run(debug=True)
