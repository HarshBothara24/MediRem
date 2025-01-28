import paddleocr
import spacy
import re
from datetime import datetime, timedelta
import json
import os
from gtts import gTTS
import gradio as gr
import subprocess
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Function to ensure SpaCy model is installed
def ensure_spacy_model(model_name):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"[INFO] Model {model_name} not found. Downloading now...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])

# Ensure the model is available
ensure_spacy_model("en_core_web_sm")

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to perform OCR on prescription image using PaddleOCR
def extract_text_from_image_paddleocr(image_path):
    try:
        print("[INFO] Starting OCR with PaddleOCR...")
        ocr = paddleocr.OCR(lang='en')  # Initialize PaddleOCR with English language
        result = ocr.ocr(image_path, cls=True)  # Extract text from image
        text = "\n".join([line[1] for line in result[0]])  # Combine text from all lines
        print(f"[DEBUG] Extracted text using PaddleOCR:\n{text}")
        return text
    except Exception as e:
        print(f"[ERROR] Error during OCR with PaddleOCR: {e}")
        return f"Error during OCR: {e}"

# Enhanced Function to extract entities (medicine, dosage, frequency, method, duration)
def extract_entities_from_text(text):
    try:
        print("[INFO] Extracting entities from text...")
        entities = {
            "medicine": [],
            "dosage": [],
            "frequency": [],
            "method": [],
            "duration": []
        }

        # Regex patterns for medicines, dosage, frequency, and duration
        medicine_matches = re.findall(r'\d+\)\s*([A-Za-z0-9\s]+?)\s*(?:\d+mg|\d+mcg|\d+ml|TAB|CAP)?', text)
        dosage_matches = re.findall(r'(\d+\s(?:Morning|Night|Afternoon|Evening|\d+\s\w+))', text, re.IGNORECASE)
        duration_matches = re.findall(r'(\d+\sDays?)', text, re.IGNORECASE)
        method_matches = re.findall(r'(Before Food|After Food|With Water)', text, re.IGNORECASE)

        # Populate entities
        entities["medicine"] = medicine_matches or ["Unidentified Medicine"]
        entities["dosage"] = dosage_matches or ["As prescribed"] * len(entities["medicine"])
        entities["duration"] = duration_matches or ["1 day"] * len(entities["medicine"])
        entities["method"] = method_matches or ["After Food"] * len(entities["medicine"])

        print(f"[DEBUG] Matches found: {entities}")
        return entities
    except Exception as e:
        print(f"[ERROR] Error during entity extraction: {e}")
        return {"error": str(e)}

# Function to parse frequency and duration into a schedule
def generate_schedule(entities):
    try:
        print("[INFO] Generating medicine schedule...")
        schedule = []

        current_time = datetime.now()
        for i, medicine in enumerate(entities["medicine"]):
            dosage = entities["dosage"][i] if i < len(entities["dosage"]) else "As prescribed"
            method = entities["method"][i] if i < len(entities["method"]) else "After Food"
            duration_text = entities["duration"][i] if i < len(entities["duration"]) else "1 day"

            duration_days = int(re.search(r"\d+", duration_text).group()) if re.search(r"\d+", duration_text) else 1

            for day in range(duration_days):
                reminder_time = current_time + timedelta(days=day)
                schedule.append({
                    "time": reminder_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "medicine": medicine,
                    "dosage": dosage,
                    "method": method
                })

        print(f"[DEBUG] Generated schedule: {schedule}")
        return schedule
    except Exception as e:
        print(f"[ERROR] Error generating schedule: {e}")
        return {"error": str(e)}

# Function to generate voice reminder using gTTS
def generate_voice_reminder(text, output_file):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
    except Exception as e:
        print(f"[ERROR] Error generating audio: {e}")

# Function to create voice reminders
def setup_voice_reminders(schedule):
    try:
        print("[INFO] Setting up voice reminders...")
        os.makedirs("reminders", exist_ok=True)
        for entry in schedule:
            reminder_text = f"It's time to take {entry['dosage']} of {entry['medicine']} {entry['method']}"
            output_file = f"reminders/reminder_{entry['time'].replace(':', '-')}.mp3"
            generate_voice_reminder(reminder_text, output_file)
    except Exception as e:
        print(f"[ERROR] Error setting up voice reminders: {e}")

# Gradio interface
def process_prescription(image):
    try:
        print("[INFO] Processing prescription image...")
        text = extract_text_from_image_paddleocr(image)
        if "Error:" in text:
            return text
        entities = extract_entities_from_text(text)
        schedule = generate_schedule(entities)
        setup_voice_reminders(schedule)
        return json.dumps(schedule, indent=4)
    except Exception as e:
        print(f"[ERROR] An error occurred during processing: {e}")
        return f"An error occurred during processing: {e}"

def main():
    interface = gr.Interface(
        fn=process_prescription,
        inputs=gr.Image(type="filepath"),
        outputs="text",
        title="Medical Prescription Reminder",
        description="Upload a prescription image to extract medicine details and generate voice reminders."
    )
    interface.launch()

if __name__ == "__main__":
    main()
