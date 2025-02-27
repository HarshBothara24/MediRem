Voice-enabled Medicine Reminder

Overview

This is a Flask-based AI-powered Medicine Reminder System that helps users extract medicine details from prescriptions using OCR (Optical Character Recognition) and schedule reminders accordingly. The system uses PaddleOCR for text extraction, PyTorch for model inference, and APScheduler for scheduling medicine reminders.

Features

Upload Prescription Image: Users can upload an image of their prescription.

OCR-Based Text Extraction: Extracts medicine names, dosages, and schedules from the image using PaddleOCR.

AI-based Classification: A PyTorch-based deep learning model analyzes the extracted text.

Automated Reminders: Schedules medicine intake reminders using APScheduler.

Supports Multiple Input Methods: Accepts image uploads via file upload or URL.

Tech Stack

Backend: Flask

Machine Learning: PyTorch, PaddleOCR

Scheduling: APScheduler

Image Processing: Pillow, torchvision

Data Handling: NumPy, regex

Installation

1. Clone the Repository

git clone https://github.com/HarshBothara24/MediRem.git
cd medicine-reminder

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies

pip install -r requirements.txt

4. Download the Pre-trained Model

Ensure that you have a trained PyTorch model named med_reminder.pth in the root directory. If not, train or download it.

Running the Application

1. Start the Flask Server

python app.py

The server will start at http://127.0.0.1:5000/.

2. API Endpoints

Home Page

GET /

Returns the web interface (if any is implemented).

Test API

GET /apidata

Returns: { "Testing Successful 200" }

Predict Medicine Details from an Image

POST /predict

Input:

file: Image file (multipart/form-data)

OR image_url: Image URL (JSON { "image_url": "URL_HERE" } or form-data)

Output: JSON with extracted medicine details.

Example Response:

{
  "extracted_text": "Paracetamol 500mg 2x 5 days",
  "extracted_medicine_data": [
    "Paracetamol 500mg 2 times a day for 5 days"
  ]
}

How It Works

User uploads a prescription image.

OCR Extraction: PaddleOCR extracts text from the image.

Regex-based Filtering: Extracts medicine names, dosages, and schedules.

Scheduling Reminders: APScheduler schedules reminders for medicines based on extracted information.

Notifications: (Optional) Can integrate with Twilio or other notification systems.


Future Enhancements

Mobile App Integration: Connect with an Android/iOS app.

Voice Commands: Enable voice-enabled interactions using Google Assistant/Alexa.

Database Storage: Save prescriptions and history in a database.

License

This project is open-source and available under the MIT License.

Author

Developed by Harsh Bothara 🚀

