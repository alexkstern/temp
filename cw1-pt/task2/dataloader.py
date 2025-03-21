import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def get_data_loaders(data_dir='./data', batch_size=64, train_subset_ratio=1.0):
    """
    Create data loaders for training and testing.
    
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for training and testing.
        train_subset_ratio (float): Fraction of the training data to use (1.0 = all data).
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full training data
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform
    )
    
    # If using only a subset of the training data, randomly sample indices.
    if train_subset_ratio < 1.0:
        subset_size = int(train_subset_ratio * len(full_train_dataset))
        indices = torch.randperm(len(full_train_dataset))[:subset_size]
        train_dataset = Subset(full_train_dataset, indices)
        print(f"Using {subset_size} out of {len(full_train_dataset)} training samples.")
    else:
        train_dataset = full_train_dataset
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Load full test data (using the entire test set)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    classes = full_train_dataset.classes
    
    return train_loader, test_loader, classes


def download_data(data_dir='./data', download=True):
    """
    Download the CIFAR-10 dataset if it doesn't exist in the specified directory.
    
    Args:
        data_dir (str): Directory to store the dataset
        download (bool): Whether to download the dataset if it doesn't exist
        
    Returns:
        bool: True if data exists or was successfully downloaded
    """
    # Check if data directory exists, create if it doesn't
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory {data_dir}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Check if CIFAR-10 dataset exists
    try:
        # Try to load the dataset to see if it exists
        torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
        print("CIFAR-10 dataset already exists.")
        return True
    except:
        if download:
            print("Downloading CIFAR-10 dataset...")
            try:
                # Download training data
                torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
                # Download test data
                torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
                print("CIFAR-10 dataset downloaded successfully.")
                return True
            except Exception as e:
                print(f"Error downloading CIFAR-10 dataset: {e}")
                return False
        else:
            print("CIFAR-10 dataset does not exist and download=False.")
            return False


def get_cifar10_loaders(data_dir='./data', batch_size=64, num_workers=2):
    """
    Create data loaders for the CIFAR-10 dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # Ensure data is downloaded
    if not download_data(data_dir):
        raise RuntimeError("Failed to download or locate the CIFAR-10 dataset.")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=False, 
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=False, 
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Get CIFAR-10 classes from the dataset
    classes = train_dataset.classes
    
    return train_loader, test_loader, classes


if __name__ == "__main__":
    # Test the functions
    print("Testing data downloading and loading...")
    download_data()
    train_loader, test_loader, classes = get_cifar10_loaders()
    
    # Display some information about the dataset
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Classes: {classes}")
    
    # Get a batch of training data
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")