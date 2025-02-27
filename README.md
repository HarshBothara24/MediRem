<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-enabled Medicine Reminder - README</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2, h3 { color: #333; }
        code { background-color: #f4f4f4; padding: 3px 5px; border-radius: 5px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
        ul { margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Voice-enabled Medicine Reminder</h1>
    <p>This is a Flask-based AI-powered Medicine Reminder system that extracts medication details from images using OCR and sets reminders based on dosage schedules.</p>
    
    <h2>Features</h2>
    <ul>
        <li>Extracts medicine details from prescriptions using PaddleOCR</li>
        <li>Schedules reminders for medications based on dosage frequency</li>
        <li>Supports image uploads and image URLs for processing</li>
        <li>Uses Flask as the backend framework</li>
        <li>Integrates PyTorch for AI-based processing</li>
    </ul>

    <h2>Installation</h2>
    <p>Follow these steps to set up and run the project:</p>
    <pre><code>git clone https://github.com/your-repo/voice-medicine-reminder.git
cd voice-medicine-reminder
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt</code></pre>

    <h2>Required Dependencies</h2>
    <p>The following libraries are used in this project:</p>
    <pre><code>
Flask
Pillow
Torch
Torchvision
numpy
paddleocr
apscheduler
requests
    </code></pre>

    <h2>Running the Application</h2>
    <p>To start the Flask server, run:</p>
    <pre><code>python app.py</code></pre>
    <p>By default, the server runs on <code>http://localhost:5000</code></p>
    
    <h2>API Endpoints</h2>
    <ul>
        <li><strong>GET /</strong> - Returns the homepage</li>
        <li><strong>POST /predict</strong> - Accepts an image file or image URL, processes the text, and extracts medicine details</li>
        <li><strong>GET /apidata</strong> - Test endpoint to check API response</li>
    </ul>
    
    <h2>Example API Request</h2>
    <p>Send a POST request with an image file:</p>
    <pre><code>curl -X POST -F "file=@prescription.jpg" http://localhost:5000/predict</code></pre>
    <p>Or send an image URL:</p>
    <pre><code>curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://example.com/image.jpg"}' http://localhost:5000/predict</code></pre>
    
    <h2>License</h2>
    <p>This project is open-source and available under the MIT License.</p>
</body>
</html>
