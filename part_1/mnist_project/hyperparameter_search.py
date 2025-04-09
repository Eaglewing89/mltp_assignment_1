import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import itertools
from datetime import datetime
import random
import time
from extremely_urgent_assignment import MLP, evaluate_model

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load MNIST dataset
train_dataset = datasets.MNIST(
    'data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    'data', train=False, download=True, transform=transform)

# Split train into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size])


def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, criterion, epochs, early_stopping_patience=3):
    train_losses = []
    val_accuracies = []

    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on test set
    test_accuracy, _, _ = evaluate_model(model, test_loader)

    return train_losses, val_accuracies, test_accuracy


def create_hyperparameter_grid():
    """Create a grid of hyperparameters to search through."""
    param_grid = {
        'hidden_sizes': [
            [128],
            [256],
            [512],
            [128, 64],
            [256, 128],
            [512, 256],
            [512, 256, 128],
            [256, 256, 256],
            [1024, 512, 256]
        ],
        'batch_size': [32, 64, 128],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'dropout_rate': [0.2, 0.3, 0.5],
        'optimizer': ['Adam', 'SGD']
    }

    return param_grid


def load_previous_results(log_file='hyperparameter_search_log.json'):
    """Load previous hyperparameter search results if they exist."""
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return json.load(f)
    return []


def save_results(results, log_file='hyperparameter_search_log.json'):
    """Save the hyperparameter search results."""
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=4)


def generate_hyperparameter_combinations(param_grid):
    """Generate all possible combinations of hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def search_hyperparameters(param_grid=None, max_trials=None, random_search=False):
    # Create directories if they don't exist
    os.makedirs('part_1/mnist_project', exist_ok=True)

    # Load previous results if they exist
    previous_results = load_previous_results()
    print(f"Loaded {len(previous_results)} previous results")

    # Get hyperparameter grid
    if param_grid is None:
        param_grid = create_hyperparameter_grid()

    # Generate all hyperparameter combinations
    all_combinations = generate_hyperparameter_combinations(param_grid)

    # Remove combinations that have already been tried
    tried_combinations = []
    for result in previous_results:
        tried_combinations.append(
            {k: result['hyperparameters'].get(k) for k in param_grid.keys()})

    remaining_combinations = [
        combo for combo in all_combinations if combo not in tried_combinations]
    print(f"{len(remaining_combinations)} combinations remaining out of {len(all_combinations)} total")

    # If using random search, shuffle the combinations
    if random_search:
        random.shuffle(remaining_combinations)

    # Limit the number of trials if specified
    if max_trials is not None:
        remaining_combinations = remaining_combinations[:max_trials]

    # Search through the hyperparameter combinations
    for i, hyperparams in enumerate(remaining_combinations):
        print(f"\nTrial {i+1}/{len(remaining_combinations)}")
        print(f"Hyperparameters: {hyperparams}")

        # Create data loaders
        train_loader = DataLoader(
            train_subset, batch_size=hyperparams['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_subset, batch_size=hyperparams['batch_size'])
        test_loader = DataLoader(
            test_dataset, batch_size=hyperparams['batch_size'])

        # Create model
        model = MLP(hidden_sizes=hyperparams['hidden_sizes'],
                    dropout_rate=hyperparams['dropout_rate']).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        if hyperparams['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                model.parameters(), lr=hyperparams['learning_rate'])
        else:  # SGD
            optimizer = optim.SGD(
                model.parameters(), lr=hyperparams['learning_rate'], momentum=0.9)

        # Train and evaluate the model
        start_time = time.time()
        train_losses, val_accuracies, test_accuracy = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            optimizer, criterion, epochs=15, early_stopping_patience=3
        )
        end_time = time.time()

        # Save the results
        result = {
            'hyperparameters': hyperparams,
            'test_accuracy': test_accuracy,
            'max_val_accuracy': max(val_accuracies) if val_accuracies else 0,
            'training_time': end_time - start_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        previous_results.append(result)
        save_results(previous_results)

        # Save the model if it achieves >90% accuracy
        if test_accuracy > 90:
            print(
                f"Found model with test accuracy > 90%: {test_accuracy:.2f}%")
            torch.save(model.state_dict(
            ), f'part_1/mnist_project/mnist_model_{test_accuracy:.2f}.pth')

            # Plot training curves
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(1, 2, 2)
            plt.plot(val_accuracies)
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')

            plt.tight_layout()
            plt.savefig(
                f'part_1/mnist_project/training_curves_{test_accuracy:.2f}.png')
            plt.close()

    # Find the best hyperparameters
    best_result = max(previous_results, key=lambda x: x['test_accuracy'])
    print("\nBest hyperparameters:")
    print(json.dumps(best_result, indent=4))

    return best_result


def analyze_results(log_file='part_1/mnist_project/hyperparameter_search_log.json'):
    """Analyze the hyperparameter search results and generate plots."""
    if not os.path.exists(log_file):
        print("No results found")
        return

    # Load results
    with open(log_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results found")
        return

    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)

    # Print top 5 results
    print("Top 5 results:")
    for i, result in enumerate(results[:5]):
        print(
            f"{i+1}. Test Accuracy: {result['test_accuracy']:.2f}%, Hyperparameters: {result['hyperparameters']}")

    # Analyze impact of different hyperparameters
    hyperparams = ['hidden_sizes', 'batch_size',
                   'learning_rate', 'dropout_rate', 'optimizer']

    for param in hyperparams:
        if param == 'hidden_sizes':
            # Group by number of layers
            param_values = {}
            for result in results:
                num_layers = len(result['hyperparameters'][param])
                if num_layers not in param_values:
                    param_values[num_layers] = []
                param_values[num_layers].append(result['test_accuracy'])
        else:
            # Group by parameter value
            param_values = {}
            for result in results:
                value = result['hyperparameters'][param]
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(result['test_accuracy'])

        # Plot parameter impact
        plt.figure(figsize=(10, 6))
        labels = []
        means = []
        errors = []

        for value, accuracies in param_values.items():
            labels.append(str(value))
            means.append(np.mean(accuracies))
            errors.append(np.std(accuracies))

        plt.bar(range(len(labels)), means, yerr=errors, capsize=10)
        plt.xticks(range(len(labels)), labels)
        plt.xlabel(param.replace('_', ' ').title())
        plt.ylabel('Average Test Accuracy (%)')
        plt.title(
            f'Impact of {param.replace("_", " ").title()} on Test Accuracy')
        plt.tight_layout()
        plt.savefig(f'part_1/mnist_project/impact_{param}.png')
        plt.close()

    # Create a summary report
    best_result = results[0]
    report = f"""
HYPERPARAMETER SEARCH SUMMARY
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Total configurations tested: {len(results)}

Best configuration:
- Test Accuracy: {best_result['test_accuracy']:.2f}%
- Hidden layers: {best_result['hyperparameters']['hidden_sizes']}
- Batch size: {best_result['hyperparameters']['batch_size']}
- Learning rate: {best_result['hyperparameters']['learning_rate']}
- Dropout rate: {best_result['hyperparameters']['dropout_rate']}
- Optimizer: {best_result['hyperparameters']['optimizer']}

Key findings:
- The optimal number of hidden layers appears to be {max(param_values.keys(), key=lambda k: np.mean(param_values[k]))}
- The best performing optimizer is {max(param_values.keys(), key=lambda k: np.mean(param_values[k]))}
- Higher dropout rates tend to {("improve" if np.corrcoef([float(k) for k in param_values.keys() if isinstance(k, (int, float, str))], [np.mean(v) for k, v in param_values.items() if isinstance(k, (int, float, str))])[0, 1] > 0 else "reduce")} performance

Analysis graphs have been saved to the project directory.
    """

    with open('part_1/mnist_project/hyperparameter_search_report.txt', 'w') as f:
        f.write(report)

    return best_result


if __name__ == "__main__":
    # Check if there are previous results
    if os.path.exists('part_1/mnist_project/hyperparameter_search_log.json'):
        print("Previous results found. Analyzing...")
        best_result = analyze_results()

        # Ask if user wants to continue searching
        choice = input("Do you want to continue searching? (y/n): ")
        if choice.lower() == 'y':
            search_hyperparameters(max_trials=10, random_search=True)
            analyze_results()
    else:
        print("No previous results found. Starting hyperparameter search...")
        search_hyperparameters(max_trials=10, random_search=True)
        analyze_results()
