#!/usr/bin/env python
# coding: utf-8

# 1. Preprocessing

# In[2]:


# get_ipython().run_line_magic('pip', 'install PIL')
# get_ipython().run_line_magic('pip', 'install torch')


# In[18]:


import os
import json
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

# Define the dataset class
class PrescriptionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, image_file.replace(".png", ".json"))

        # Load the image
        try:
            image = Image.open(image_path).convert("RGB")  # Ensures all images are in RGB format
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            return None  # Skip this image if it can't be loaded

        # Apply transformations (resize, convert to tensor)
        if self.transform:
            image = self.transform(image)

        # Load the corresponding label
        try:
            with open(label_path, 'r') as f:
                label = json.load(f)
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return None  # Skip this image-label pair if the label can't be loaded

        # Convert the label into a tensor (assuming it's a dictionary with numerical data)
        label_tensor = self.convert_label_to_tensor(label)

        return image, label_tensor

    def convert_label_to_tensor(self, label):
        # Assuming label is a dictionary of numerical values (e.g., dose, frequency)
        # You need to adjust this based on the actual structure of your label
        try:
            label_values = [v for v in label.values()]  # Get all values from the dictionary
            label_tensor = torch.tensor(label_values, dtype=torch.float32)  # Convert to tensor
            return label_tensor
        except Exception as e:
            print(f"Error converting label to tensor: {e}")
            return torch.zeros(1)  # Return a tensor of zeros if conversion fails

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256 to ensure uniformity
    transforms.ToTensor(),  # Convert to tensor
])

# Specify image and label directories
images_dir = 'dataset/images'
labels_dir = 'dataset/labels'

# Define a custom collate function to handle batching
def custom_collate_fn(batch):
    # Filter out None values from the batch (i.e., skipped corrupt images or labels)
    batch = [item for item in batch if item is not None]

    # Check if the batch is empty after filtering (i.e., all items were skipped)
    if len(batch) == 0:
        return None, None

    # Separate images and labels
    images, labels = zip(*batch)

    # Stack images and labels into tensors
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels

# Create the dataset and dataloaders
dataset = PrescriptionDataset(images_dir, labels_dir, transform=transform)

# Create DataLoader with custom collate function
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

# Check how data is being loaded in batches
try:
    for batch_idx, (images, labels) in enumerate(train_loader):
        if images is None:
            print("Warning: Batch contains only invalid data.")
            continue  # Skip empty batches
        print(f"Batch {batch_idx} - Image batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"First image shape in the batch: {images[0].shape}")

except RuntimeError as e:
    print(f"Error occurred: {e}")


# 2. Model Architecture

# In[7]:


import torch
import torch.nn as nn
from torchvision import models

# Define the model architecture
class PrescriptionModel(nn.Module):
    def __init__(self):
        super(PrescriptionModel, self).__init__()
        # Load the pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the fully connected layer to suit the output
        # Let's assume you need to predict 5 different attributes for the medicine (e.g., name, dosage, method, frequency, duration)
        self.fc = nn.Linear(self.resnet.fc.in_features, 5)

    def forward(self, x):
        # Pass through ResNet layers
        x = self.resnet(x)
        x = self.fc(x)
        return x

# Initialize the model
model = PrescriptionModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Assuming classification for each attribute
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 3. Training the model

# In[8]:


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Move data to GPU if available
        images, labels = images.cuda(), labels.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")


# In[1]:


import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define a custom dataset class
class OCRDataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_list, transform=None):
        """
        :param images_dir: Directory containing images
        :param labels_dir: Directory containing labels as individual JSON files
        :param image_list: List of image file names
        :param transform: Transformations to apply to images
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, image_name.replace('.png', '.json'))  # Adjust if using different extension

        # Load the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load and process the label
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Extract medicines and syrups counts
        medicines_count = len(label_data.get('medicines', []))
        syrups_count = len(label_data.get('syrups', []))
        
        # Use counts as labels (you can modify this logic for your task)
        label = torch.tensor([medicines_count, syrups_count], dtype=torch.float32)

        return image, label

# Define the model architecture
class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust dimensions for your dataset
        self.fc2 = nn.Linear(128, 2)  # Two outputs: medicines count and syrups count

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Split dataset into train and test
def split_dataset(images_dir, test_size=0.2):
    image_names = [f for f in os.listdir(images_dir) if f.endswith('.png')]  # Adjust if images have a different extension
    train_images, test_images = train_test_split(image_names, test_size=test_size, random_state=42)
    return train_images, test_images

# Main script
def main():
    images_dir = 'dataset/images'  # Directory containing images
    labels_dir = 'dataset/labels'  # Directory containing labels JSON files
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Split the dataset
    train_images, test_images = split_dataset(images_dir)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Create datasets
    train_dataset = OCRDataset(images_dir, labels_dir, train_images, transform)
    test_dataset = OCRDataset(images_dir, labels_dir, test_images, transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = OCRModel()
    criterion = nn.MSELoss()  # Mean Squared Error for regression task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Testing loop
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader)}")

    # Save the model
    torch.save(model.state_dict(), "med_reminder.pth")
    print("Model saved as med_reminder.pth")

if __name__ == "__main__":
    main()


# In[2]:


import torch

# Load your model architecture and weights
model = OCRModel()  # Replace with your model class
model.load_state_dict(torch.load("med_reminder.pth"))
model.eval()  # Set to evaluation mode


# In[ ]:


from PIL import Image
import torchvision.transforms as transforms

# Load an unseen prescription image
image_path = "D:/Coding/Machine_Learning/Projects/voice_assisted_medicine_reminder/dataset/test_prescriptions/1.png"  # Replace with your image path
image = Image.open(image_path).convert("RGB")

# Apply the same transformations as during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Preprocess the image
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension


# In[6]:


# Move the model to GPU
model = model.cuda()  # Move model to GPU


# In[7]:


with torch.no_grad():
    # Move image tensor to GPU
    image_tensor = image_tensor.cuda()  # Ensure image is on the GPU
    # Forward pass through the model
    output = model(image_tensor)  # Get predictions

    # Move output back to CPU for further processing
    predictions = output.cpu().numpy()  # Convert to NumPy array
    print("Raw Predictions:", predictions)


# Testing

# In[9]:


import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class UnseenDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Create the dataset and DataLoader
unseen_data_dir = "dataset/test_prescriptions"
unseen_dataset = UnseenDataset(image_dir=unseen_data_dir, transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=1, shuffle=False)


# In[10]:


import torch

model = OCRModel()  # Replace with your model definition
model.load_state_dict(torch.load("med_reminder.pth"))
model.eval()
model.cuda()  # Move to GPU if available


# In[15]:


def process_model_output(output):
    # Example: Convert tensor output into structured data
    medicines = []  # Parse medicine details
    syrups = []     # Parse syrup details

    for pred in output:
        # Assuming `output` contains structured prediction data
        details = {
            "name": pred["name"],
            "dosage": pred["dosage"],
            "frequency": pred["frequency"],
            "duration": pred["duration"]
        }
        if pred["type"] == "medicine":
            medicines.append(details)
        elif pred["type"] == "syrup":
            syrups.append(details)
    
    return {"medicines": medicines, "syrups": syrups}


# In[16]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[20]:


from PIL import Image
import torchvision.transforms as transforms

# Define the transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Placeholder for storing results
results = []

# Loop through unseen test data
unseen_data_path = "dataset/test_prescriptions"  # Path to unseen data images
image_files = ["1.png", "2.png"]  # Replace with dynamic listing of files

for img_file in image_files:
    # Load and preprocess the image
    image_path = f"{unseen_data_path}/{img_file}"
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    

    # Inference
    with torch.no_grad():
        output = model(image)  # Get predictions
        # Post-process the output
        predicted_medicines = process_model_output(output)  # Custom function for processing
        
        # Save the predictions for this image
        results.append({
            "image_file": img_file,
            "predictions": predicted_medicines
        })

        print(f"Processed {img_file}: {predicted_medicines}")


# In[ ]:




