import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Step 1: Load and Normalize the CIFAR-100 Dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5074, 0.4867, 0.4411), (0.2675, 0.2565, 0.2761))
])

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Step 2: Define the CNN Model (Same architecture as before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3: Load the Trained Model
net = SimpleCNN()

# device = "cuda"
# net = net.to(device)

net.load_state_dict(torch.load("cifar100_cnn.pth"))

# Step 4: Define the Loss Function
criterion = nn.CrossEntropyLoss()

# Step 5: Evaluate the Model and Calculate the Loss
net.eval()  # Set the model to evaluation mode

total_loss = 0.0
total_correct = 0
total_samples = 0

with torch.no_grad():  # Disable gradient computation for inference
    for data in testloader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)  # Accumulate the loss
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

average_loss = total_loss / total_samples
accuracy = 100 * total_correct / total_samples

print(f"Average loss on the test set: {average_loss:.4f}")
print(f"Accuracy on the test set: {accuracy:.2f}%")
