import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a non-linearly separable 3D dataset


def create_dataset(n_samples=1000):
    # Create two classes with some overlap
    class1 = np.random.randn(n_samples//2, 3) * 0.5
    class2 = np.random.randn(n_samples//2, 3) * 0.5

    # Add some non-linear transformation to make it interesting
    class1[:, 0] = class1[:, 0] + np.sin(class1[:, 1] * 2) * 0.5
    class2[:, 0] = class2[:, 0] - np.sin(class2[:, 1] * 2) * 0.5

    # Shift class2
    class2 = class2 + np.array([1.0, 0.5, 0.3])

    X = np.vstack((class1, class2))
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))

    return X, y

# Define the MLP model


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(3, 5)  # 3 input features, 5 hidden neurons
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(5, 2)  # 5 hidden neurons, 2 output classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.leaky_relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x


# Create dataset
X, y = create_dataset(1000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Initialize model, loss function, and optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01,
                       weight_decay=0.01)  # L2 regularization
scheduler = ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.5)  # Learning rate scheduler

# Early stopping parameters
patience = 10
min_delta = 0.001
best_val_loss = float('inf')
counter = 0

# Training loop
n_epochs = 200
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Store losses for plotting
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Calculate final test accuracy
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Final Test Accuracy: {accuracy:.4f}")

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.savefig('training_losses.png')
plt.close()

# Plot the 3D dataset
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='r', label='Class 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='b', label='Class 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Dataset Visualization')
plt.legend()
plt.savefig('dataset_visualization.png')
plt.close()
