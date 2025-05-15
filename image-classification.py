import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Select device: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Define transformations: convert images to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load training dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=0)

# Download and load test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)

# Define class names for CIFAR-10
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a simple CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: input 3 channels, output 16 channels, kernel 3x3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Max pooling layer with 2x2 window
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 16 input channels, 32 output channels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 8x8 is the image size after pooling twice
        self.fc2 = nn.Linear(128, 10)          # 10 output classes

    def forward(self, x):
        # Pass through first conv layer, apply ReLU, then pool
        x = self.pool(F.relu(self.conv1(x)))
        # Pass through second conv layer, ReLU, then pool again
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for fully connected layers
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))  # Apply ReLU after first FC
        x = self.fc2(x)          # Final layer outputs raw scores (logits)
        return x

# Instantiate model and move to device
net = Net().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()                   # Suitable for multi-class classification
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD with momentum

# Training parameters
num_epochs = 5
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()  # Set model to training mode
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()           # Zero the parameter gradients
        outputs = net(inputs)           # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate and save epoch loss and accuracy
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')

print('Training completed.')

# Evaluate model on test data
all_preds = []
all_labels = []

net.eval()  # Set model to evaluation mode
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=classes))

# Plot training loss and accuracy curves
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy', color='green')
plt.title('Training Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Helper function to display an image with title
def imshow(img, title):
    img = img / 2 + 0.5   # Unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Show some test images with predictions
dataiter = iter(testloader)
images, labels = next(dataiter)
net.eval()
with torch.no_grad():
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)

for i in range(5):
    imshow(images[i], f'True: {classes[labels[i]]} / Predicted: {classes[predicted[i]]}')

# Collect wrongly predicted images for inspection
wrong_images = []
wrong_labels = []
wrong_preds = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0], data[1]
        outputs = net(images.to(device))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
        wrong_mask = preds != labels
        if wrong_mask.any():
            wrong_images.append(images[wrong_mask])
            wrong_labels.append(labels[wrong_mask])
            wrong_preds.append(preds[wrong_mask])

wrong_images = torch.cat(wrong_images)
wrong_labels = torch.cat(wrong_labels)
wrong_preds = torch.cat(wrong_preds)

print(f"Total wrong predictions: {wrong_images.shape[0]}")

# Show some wrongly classified images
for i in range(5):
    imshow(wrong_images[i], f'❌ True: {classes[wrong_labels[i]]} / Predicted: {classes[wrong_preds[i]]}')
