import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def get_data_loaders(data_dir='./data', batch_size=64):
    """
    Create data loaders for training and testing.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training and testing
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get classes from the dataset
    classes = train_dataset.classes
    
    return train_loader, test_loader, classes

def fit_elm_sgd(model, train_loader, test_loader, num_epochs=10, 
                learning_rate=0.01, mixup=None, save_path=None, mixup_save_path='mixup.png'):
    """
    Train an ELM model using Stochastic Gradient Descent.
    
    Args:
        model (nn.Module): The ELM model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for SGD
        mixup (MyMixUp, optional): MixUp augmentation object
        save_path (str, optional): Path to save the trained model
        mixup_save_path (str, optional): Path to save mixup examples
        
    Returns:
        dict: Training results including losses and metrics
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    if mixup is not None:
        # For MixUp, we need a criterion that works with soft labels (one-hot encoded)
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Dictionary to store training results
    results = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    # For saving MixUp samples
    if mixup is not None:
        mixup_samples = []
    
            # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(train_loader, 0):
            if i % 20 == 0:  # Print progress every 20 batches
                print(f"  Batch {i}/{len(train_loader)}", end="\r")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply MixUp if provided
            if mixup is not None:
                # Generate mixed samples
                mixed_inputs, mixed_labels = mixup(inputs, labels)
                
                # Save some mixup examples for visualization (once per training)
                if len(mixup_samples) < 16 and i == 0:
                    print(f"Saving MixUp examples...")
                    for j in range(min(16, inputs.size(0))):
                        # Save original and mixed versions
                        if len(mixup_samples) < 8:
                            mixup_samples.append(inputs[j].cpu())
                        mixup_samples.append(mixed_inputs[j].cpu())
                
                # Use mixed inputs and labels for training
                inputs = mixed_inputs
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss with one-hot encoded labels from mixup
                loss = criterion(outputs, mixed_labels)
            else:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy for regular (non-mixup) batches
            if mixup is None or epoch == 0:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average training loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total if (mixup is None or epoch == 0) else 0.0  # Skip acc calculation for mixup
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Store results
        results['train_loss'].append(epoch_loss)
        results['test_loss'].append(test_loss)
        results['train_accuracy'].append(epoch_acc)
        results['test_accuracy'].append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {test_acc:.2f}%')
    
    # Save mixup examples if available
    if mixup is not None and len(mixup_samples) > 0:
        save_image(torch.stack(mixup_samples), 'mixup.png', nrow=4, normalize=True)
    
    # Save model if path is provided
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    # Save mixup examples if available
    if mixup is not None and len(mixup_samples) > 0 and mixup_save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(mixup_save_path) or '.', exist_ok=True)
            
            # Save images - include both original and mixed versions
            save_image(torch.stack(mixup_samples), mixup_save_path, nrow=4, normalize=True)
            print(f"MixUp examples saved to {mixup_save_path}")
        except Exception as e:
            print(f"Error saving mixup examples: {e}")
            import traceback
            print(traceback.format_exc())
    
    return results

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader for evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    print("Evaluating model on test set...")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i % 20 == 0:  # Print progress every 20 batches
                print(f"  Eval Batch {i}/{len(data_loader)}", end="\r")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    average_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return average_loss, accuracy

def save_results(model, test_loader, classes, save_path='result.png'):
    """
    Save visualization of model predictions on test data.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): DataLoader for test data
        classes (tuple): Class names
        save_path (str): Path to save the visualization
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get batch of images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Get first 36 images
    images = images[:36].cpu()
    labels = labels[:36].cpu()
    predicted = predicted[:36].cpu()
    
    # Save images
    save_image(images, save_path, nrow=6, normalize=True)
    
    # Print ground truth and predictions
    print("Ground Truth and Predictions:")
    for i in range(36):
        print(f"Image {i+1}: Ground Truth = {classes[labels[i]]}, Predicted = {classes[predicted[i]]}")