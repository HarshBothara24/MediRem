import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pytesseract
import json
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset class
class MedReminderDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".json")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load or extract labels
        with open(label_path, "r") as f:
            labels = json.load(f)

        return image, labels

    @staticmethod
    def extract_prescription_details(image_path):
        """
        Extract text from the image using OCR (pytesseract).
        """
        text = pytesseract.image_to_string(Image.open(image_path))
        return text

    @staticmethod
    def parse_prescription_details(extracted_text):
        """
        Parse extracted text to identify medicines and syrups with their details.
        Uses regex for improved accuracy.
        """
        medicines = []
        syrups = []
        
        # Define regex patterns
        dosage_pattern = r"(\d+(\.\d+)?\s?(mg|g|ml|mcg|unit))"
        frequency_pattern = r"(\d+\s*(x|times)?\s*(per\s*day|daily|once|twice|\d+\s*times))"
        duration_pattern = r"(\d+\s*(days?|weeks?|months?))"
        
        # Medicine and syrup keywords (using regex)
        medicine_keywords = r"(tablet|tab|cap|capsule|pill|medicine)"
        syrup_keywords = r"(syrup|syp|liquid)"
        
        lines = extracted_text.split("\n")
        for line in lines:
            line = line.strip()

            # Skip lines with irrelevant information
            if any(irrelevant in line.lower() for irrelevant in ["dr.", "address", "phone", "signature"]):
                continue

            # Check for medicines using regex
            if re.search(medicine_keywords, line.lower()):
                medicine = MedReminderDataset.parse_line(line, dosage_pattern, frequency_pattern, duration_pattern, "medicine")
                if medicine:
                    medicines.append(medicine)
            
            # Check for syrups using regex
            elif re.search(syrup_keywords, line.lower()):
                syrup = MedReminderDataset.parse_line(line, dosage_pattern, frequency_pattern, duration_pattern, "syrup")
                if syrup:
                    syrups.append(syrup)

        return medicines, syrups

    @staticmethod
    def parse_line(line, dosage_pattern, frequency_pattern, duration_pattern, type_of_med):
        """
        Parse a single line to extract details like name, dosage, frequency, and duration using regex.
        Type of medicine or syrup is passed as an argument ('medicine' or 'syrup').
        """
        details = {"name": "", "dosage": "", "frequency": "", "duration": "", "type": type_of_med}
        
        # Extract the name (first word in the line as name assumption)
        tokens = line.split()
        if len(tokens) > 0:
            details["name"] = tokens[0]  # First token is typically the name

        # Use regex to extract dosage, frequency, and duration
        dosage_match = re.search(dosage_pattern, line)
        if dosage_match:
            details["dosage"] = dosage_match.group(0)

        frequency_match = re.search(frequency_pattern, line)
        if frequency_match:
            details["frequency"] = frequency_match.group(0)

        duration_match = re.search(duration_pattern, line)
        if duration_match:
            details["duration"] = duration_match.group(0)

        # If the line contains relevant information, return the details
        if any(details[key] != "" for key in details):
            return details
        return None

# Custom collate function for DataLoader
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])  # Stack all images
    labels = [item[1] for item in batch]  # Keep labels as-is (list of dictionaries)
    return images, labels

# Model definition
class MedReminderModel(nn.Module):
    def __init__(self, num_classes):
        super(MedReminderModel, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Loss calculation (customized for dictionary labels)
def calculate_loss(outputs, labels, criterion):
    total_loss = 0.0
    for i, label_dict in enumerate(labels):
        medicines = label_dict.get("medicines", [])
        syrups = label_dict.get("syrups", [])
        loss = criterion(outputs[i], torch.tensor(len(medicines) + len(syrups), dtype=torch.float32))
        total_loss += loss
    return total_loss / len(labels)

# Main function
def main():
    image_dir = "dataset/images"
    label_dir = "dataset/labels"
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 10  # Adjust based on your classification categories

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = MedReminderDataset(image_dir, label_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Model, criterion, and optimizer
    model = MedReminderModel(num_classes=num_classes)
    criterion = nn.MSELoss()  # Example criterion; adapt for your specific use case
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = calculate_loss(outputs, labels, criterion)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Save the model
    torch.save(model.state_dict(), "med_reminder.pth")
    print("Model saved as med_reminder.pth")

if __name__ == "__main__":
    main()


# In[ ]:




