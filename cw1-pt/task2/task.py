import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Import custom modules
from dataloader import download_data, get_data_loaders
from model import MyExtremeLearningMachine, MyMixUp, MyEnsembleELM
from train import fit_elm_sgd, evaluate, save_results
import math

# Create task directory if it doesn't exist
os.makedirs('task2', exist_ok=True)

"""
What constitutes a "random guess" in multiclass classification and how can it be tested?

In multiclass classification with balanced classes like CIFAR-10 (10 classes), a random guess 
would mean assigning each sample to a random class with equal probability (1/10 = 10%). 
The expected accuracy of random guessing is 1/number_of_classes, or 10% for CIFAR-10. 
To test this, we can create a "dummy" classifier that predicts random classes and 
evaluate its performance on the test set. Any useful model should perform significantly 
better than this random baseline.
"""

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)

# Define hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 3  # Use a smaller number of epochs for initial testing
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128

KERNEL_SIZE = 5
#STD_DEV = 0.1#1 TEST THIS
STD_DEV=0.05# BEST ONE

NUM_MODELS = 5
MIXUP_ALPHA = 4.0#0.2#2#1.0
MIXUP_PROB = 1.0#0.5  # Apply MixUp to 50% of batches

#Smaller weights yield smaller activations in the convolutional layer. 
# This can stabilize training since the fully connected layer then receives more modestly scaled features. 
# However, if the weights are too small, the resulting features might lack sufficient diversity, 
# which could limit the richness of representations the model can extract. 

# You can adjust this for quicker testing or full training
TESTING_MODE = True  # Set to False for the final run
if TESTING_MODE:
    print("RUNNING IN TESTING MODE WITH REDUCED EPOCHS")
    NUM_EPOCHS = 1  # Just 1 epoch for testing
    NUM_MODELS = 2  # Fewer models in ensemble for testing

# Download the dataset if needed
download_success = download_data()
if not download_success:
    print("Failed to download data. Exiting.")
    exit(1)

# Get data loaders
train_subset_ratio=0.2
train_loader, test_loader, classes = get_data_loaders(batch_size=BATCH_SIZE, train_subset_ratio=train_subset_ratio)

print(classes)
# Create directory to save models
os.makedirs('task2/models', exist_ok=True)

# Define criterion for evaluation
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_and_evaluate(model_name, model, use_mixup=False):
    """Train and evaluate a model with given configuration."""
    print(f"\n=== Training {model_name} ===")
    
    # Initialize MixUp if needed
    mixup = MyMixUp(alpha=MIXUP_ALPHA, seed=SEED) if use_mixup else None
    
    # Create directory structure if needed
    os.makedirs('task2/models', exist_ok=True)
    
    # Set paths
    model_save_path = f"task2/models/{model_name}.pth"
    
    # For MixUp models, set the appropriate path for visualization
    if use_mixup:
        print(f"Using MixUp augmentation with alpha={MIXUP_ALPHA}")
        mixup_save_path = os.path.join('task2', f"mixup_{model_name}.png")
    else:
        mixup_save_path = None
    
    # Train the model
    results = fit_elm_sgd(
        model, train_loader, test_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        mixup=mixup,
        save_path=model_save_path,
        mixup_save_path=mixup_save_path
    )
    
    # Print final performance
    print(f"\n{model_name} Final Performance:")
    print(f"Test Loss: {results['test_loss'][-1]:.4f}")
    print(f"Test Accuracy: {results['test_accuracy'][-1]:.2f}%")
    
    return results

def load_and_evaluate(model_name, model):
    """Load a saved model and evaluate it."""
    model_path = f"task2/models/{model_name}.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        
        # Evaluate the model
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        print(f"\n{model_name} Loaded Performance:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        return test_loss, test_accuracy
    else:
        print(f"Model file {model_path} not found.")
        return None, None

def main():
    """Main function to run experiments."""
    print("\n=== COMP0197 CW1 Task 2: Regularising Extreme Learning Machines ===")
    
    # Experiment 1: Basic ELM (without regularization)
    basic_elm = MyExtremeLearningMachine(
        hidden_size=HIDDEN_SIZE,
        std_dev=STD_DEV,
        kernel_size=KERNEL_SIZE,
        seed=SEED
    )
    basic_results = train_and_evaluate("basic_elm", basic_elm)
    
    # Experiment 2: ELM with MixUp
    mixup_elm = MyExtremeLearningMachine(
        hidden_size=HIDDEN_SIZE,
        std_dev=STD_DEV,
        kernel_size=KERNEL_SIZE,
        seed=SEED
    )
    mixup_results = train_and_evaluate("mixup_elm", mixup_elm, use_mixup=True)
    
    # Experiment 3: ELM with Ensemble
    ensemble_elm = MyEnsembleELM(
        hidden_size=HIDDEN_SIZE,
        std_dev=STD_DEV,
        kernel_size=KERNEL_SIZE,
        num_models=NUM_MODELS,
        seed=SEED
    )
    ensemble_results = train_and_evaluate("ensemble_elm", ensemble_elm)
    
    # Experiment 4: ELM with both MixUp and Ensemble
    ensemble_mixup_elm = MyEnsembleELM(
        hidden_size=HIDDEN_SIZE,
        std_dev=STD_DEV,
        kernel_size=KERNEL_SIZE,
        num_models=NUM_MODELS,
        seed=SEED
    )
    ensemble_mixup_results = train_and_evaluate("ensemble_mixup_elm", ensemble_mixup_elm, use_mixup=True)
    
    """
    Performance Metrics Justification:
    1. Accuracy: Direct measure of correct predictions, intuitive and relevant for balanced
       classification tasks like CIFAR-10 where all classes are equally important.
    2. Loss: Cross-entropy loss indicates confidence of predictions and is differentiable,
       making it suitable for gradient-based optimization of neural networks.
    """
    
    # Make sure we save files to the task2 directory
    mixup_path = os.path.join('task2', 'mixup.png')
    result_path = os.path.join('task2', 'result.png')
    
    # Determine the best model based on test accuracy
    models = {
        "Basic ELM": basic_results['test_accuracy'][-1],
        "MixUp ELM": mixup_results['test_accuracy'][-1],
        "Ensemble ELM": ensemble_results['test_accuracy'][-1],
        "Ensemble+MixUp ELM": ensemble_mixup_results['test_accuracy'][-1]
    }
    
    best_model_name = max(models, key=models.get)
    best_accuracy = models[best_model_name]
    
    print(f"\n=== Best Model: {best_model_name} with Test Accuracy: {best_accuracy:.2f}% ===")
    
    # Visualize results for the best model
    if best_model_name == "Basic ELM":
        model = basic_elm
    elif best_model_name == "MixUp ELM":
        model = mixup_elm
    elif best_model_name == "Ensemble ELM":
        model = ensemble_elm
    else:  # "Ensemble+MixUp ELM"
        model = ensemble_mixup_elm
    
    # Save results from best model
    save_results(model, test_loader, classes, save_path=result_path)

if __name__ == "__main__":
    main()