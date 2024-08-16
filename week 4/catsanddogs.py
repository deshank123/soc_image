import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    # Resize images to 64x64 pixels
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with given mean and std
])

# Custom dataset class to handle image loading errors
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.failed_images = []  # List to store paths of images that fail to load

    def __getitem__(self, index):
        try:
            return super(CustomImageFolder, self).__getitem__(index)  # Try to load the image
        except (IOError, OSError, ValueError) as e:
            self.failed_images.append(self.samples[index])  # If loading fails, store the image path
            return None  # Return None for failed images

# Custom collate function to filter out None values (failed images)
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))  # Remove None values from the batch
    return torch.utils.data.dataloader.default_collate(batch)  # Use default collate function for remaining items

# Define the directory for the dataset
data_dir = '/kaggle/input/microsoft-catsvsdogs-dataset/PetImages'  # Update this path to your dataset location

# Load the dataset
full_dataset = CustomImageFolder(root=data_dir, transform=transform)

# Filter out failed images from the dataset
full_dataset.samples = [sample for sample in full_dataset.samples if sample not in full_dataset.failed_images]

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))  # 80% of the data for training
validation_size = len(full_dataset) - train_size  # Remaining 20% for validation
train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Max pooling layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer with 2 classes
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization
        self.relu = nn.ReLU()  # ReLU activation function
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for output layer

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Apply first conv layer, ReLU, and max pooling
        x = self.pool(self.relu(self.conv2(x)))  # Apply second conv layer, ReLU, and max pooling
        x = x.view(-1, 64 * 16 * 16)  # Flatten the tensor for fully connected layer
        x = self.relu(self.fc1(x))  # Apply first fully connected layer and ReLU
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply output layer
        return self.softmax(x)  # Apply softmax to get class probabilities

# Instantiate the model
model = SimpleCNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training loop parameters
num_epochs = 10  # Number of epochs to train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise CPU
model.to(device)  # Move the model to the device (GPU/CPU)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass: compute model outputs
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
        
        running_loss += loss.item()  # Accumulate loss

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')  # Print average loss for the epoch

print('Training finished')  # Training complete

# Evaluation on validation set
model.eval()  # Set model to evaluation mode
correct = 0  
total = 0  # Initialize total predictions counter and correct prediction counter

with torch.no_grad():  # Disable gradient calculation for evaluation
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the device
        outputs = model(inputs)  # Forward pass: compute model outputs
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class by finding max probability
        total += labels.size(0)  # Increment total predictions count
        correct += (predicted == labels).sum().item()  # Increment correct predictions count

# evaluate the model, Calculate and print validation accuracy
print(f' Accuracy of validation: {100 * correct / total}%')