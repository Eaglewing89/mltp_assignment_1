import numpy as np


class ann_layer:
    def __init__(self, n_neurons, input_dim, use_softmax=False):
        """
        Initialize a neural network layer.

        Args:
            n_neurons (int): Number of neurons in the layer
            input_dim (int): Dimension of the input
            use_softmax (bool): Whether to use softmax activation for output
        """
        # He initialization
        self.weights = np.random.randn(
            n_neurons, input_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((n_neurons, 1))
        self.use_softmax = use_softmax

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=0))
        return exp_x / np.sum(exp_x, axis=0)

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x (np.ndarray): Input vector or matrix of shape (input_dim,) or (input_dim, batch_size)

        Returns:
            np.ndarray: Output of the layer
        """
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Linear transformation
        z = np.dot(self.weights, x) + self.biases

        # Apply ReLU
        a = self.relu(z)

        # Apply softmax if specified
        if self.use_softmax:
            a = self.softmax(a)

        return a


# Create a dataset of 10 2D points
np.random.seed(42)  # For reproducibility
dataset = np.random.randn(2, 10)  # 2 features, 10 samples

# Create the neural network layers
layer1 = ann_layer(n_neurons=3, input_dim=2)  # First hidden layer
layer2 = ann_layer(n_neurons=3, input_dim=3)  # Second hidden layer
layer3 = ann_layer(n_neurons=2, input_dim=3, use_softmax=True)  # Output layer

# Forward pass through the network
print("Input dataset shape:", dataset.shape)
print("\nDataset:")
print(dataset)

# Process through each layer
output1 = layer1.forward(dataset)
print("\nLayer 1 output shape:", output1.shape)
print("Layer 1 output:")
print(output1)

output2 = layer2.forward(output1)
print("\nLayer 2 output shape:", output2.shape)
print("Layer 2 output:")
print(output2)

final_output = layer3.forward(output2)
print("\nFinal output shape:", final_output.shape)
print("Final output (with softmax):")
print(final_output)
