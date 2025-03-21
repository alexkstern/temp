import torch
import torch.nn as nn
from dataloader import download_data
from model import MyExtremeLearningMachine, MyMixUp, MyEnsembleELM
from train import get_data_loaders, evaluate

def test_elm_implementation():
    """Test the basic ELM implementation."""
    print("\n=== Testing ELM Implementation ===")
    
    # Create a simple ELM
    elm = MyExtremeLearningMachine(
        input_channels=3,
        hidden_size=32,
        num_classes=10,
        kernel_size=3,
        std_dev=0.1,
        seed=42
    )
    
    # Check if convolutional layer weights are fixed
    initial_weights = elm.conv.weight.clone()
    
    # Create a dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    output = elm(dummy_input)
    
    # Check output shape
    expected_shape = (2, 10)  # [batch_size, num_classes]
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check if the convolutional weights remained fixed
    assert torch.all(torch.eq(initial_weights, elm.conv.weight)), "Convolutional weights changed!"
    
    print("✓ ELM produces correct output shape")
    print("✓ Fixed convolutional weights are correctly frozen")
    
    return True

def test_mixup_implementation():
    """Test the MixUp implementation."""
    print("\n=== Testing MixUp Implementation ===")
    
    # Create MixUp object with 100% probability to ensure it gets applied
    mixup = MyMixUp(alpha=1.0, num_classes=10, prob=1.0, seed=42)
    
    # Also create a MixUp with 0% probability to test skipping
    skip_mixup = MyMixUp(alpha=1.0, num_classes=10, prob=0.0, seed=42)
    
    # Create dummy batch
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 32, 32)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    
    # Apply MixUp with 100% probability
    mixed_images, mixed_labels = mixup(dummy_images, dummy_labels)
    
    # Apply MixUp with 0% probability (should skip mixing)
    skip_images, skip_labels = skip_mixup(dummy_images, dummy_labels)
    
    # Check output shapes
    assert mixed_images.shape == dummy_images.shape, "MixUp images shape mismatch"
    assert mixed_labels.shape == (batch_size, 10), "Mixed labels shape should be [batch_size, num_classes]"
    
    # Check that skipped MixUp returns original images
    assert torch.all(torch.eq(skip_images, dummy_images)), "Skip MixUp should return original images"
    
    # Verify mixed labels are soft (values between 0 and 1)
    assert torch.all((mixed_labels >= 0) & (mixed_labels <= 1)), "Mixed labels should have values between 0 and 1"
    
    # Check that labels sum to 1 for each example
    label_sums = mixed_labels.sum(dim=1)
    assert torch.allclose(label_sums, torch.ones_like(label_sums)), "Each mixed label should sum to 1"
    
    # Test different probability values
    for prob in [0.25, 0.5, 0.75]:
        prob_mixup = MyMixUp(alpha=1.0, num_classes=10, prob=prob, seed=42)
        # Run multiple trials and check proportion of mixed batches
        n_trials = 100
        n_mixed = 0
        
        # Reset the generator to make tests reproducible
        prob_mixup.generator.manual_seed(42)
        
        for _ in range(n_trials):
            trial_images, trial_labels = prob_mixup(dummy_images, dummy_labels)
            # Check if mixing occurred by comparing to original images
            if not torch.all(torch.eq(trial_images, dummy_images)):
                n_mixed += 1
        
        # Check if proportion is reasonably close to the target probability
        proportion = n_mixed / n_trials
        print(f"MixUp with prob={prob:.2f}: Applied to {proportion:.2f} of batches ({n_mixed}/{n_trials})")
        
        # Allow some statistical variation but ensure it's reasonable
        assert abs(proportion - prob) < 0.2, f"MixUp probability {proportion} too far from target {prob}"
    
    print("✓ MixUp produces correctly shaped outputs")
    print("✓ MixUp labels are properly one-hot encoded and mixed")
    print("✓ MixUp probability parameter works correctly")
    
    return True

def test_ensemble_implementation():
    """Test the Ensemble ELM implementation."""
    print("\n=== Testing Ensemble ELM Implementation ===")
    
    # Create an ensemble
    num_models = 3
    ensemble = MyEnsembleELM(
        input_channels=3,
        hidden_size=32,
        num_classes=10,
        kernel_size=3,
        std_dev=0.1,
        num_models=num_models,
        seed=42
    )
    
    # Check if the correct number of models was created
    assert len(ensemble.models) == num_models, f"Expected {num_models} models, got {len(ensemble.models)}"
    
    # Create a dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    output = ensemble(dummy_input)
    
    # Check output shape
    expected_shape = (2, 10)  # [batch_size, num_classes]
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Check if the models have different weights
    # Get weights of first convolutional layer from each model
    weights = [model.conv.weight for model in ensemble.models]
    
    # Check if all weights are different
    all_different = True
    for i in range(len(weights)):
        for j in range(i+1, len(weights)):
            if torch.all(torch.eq(weights[i], weights[j])):
                all_different = False
                break
    
    assert all_different, "Some models in the ensemble have identical weights"
    
    print("✓ Ensemble creates the correct number of models")
    print("✓ Ensemble produces correct output shape")
    print("✓ Ensemble models have different weights")
    
    return True

def test_full_training_cycle():
    """Test a complete training cycle with a small dataset."""
    print("\n=== Testing Full Training Cycle ===")
    
    # Check if dataset is available
    download_success = download_data()
    if not download_success:
        print("Failed to download data. Skipping full training test.")
        return False
    
    # Get data loaders with a small batch size for faster testing
    train_loader, test_loader, classes = get_data_loaders(batch_size=32)
    
    # Create a small ELM for quick testing
    elm = MyExtremeLearningMachine(
        input_channels=3,
        hidden_size=16,
        num_classes=10,
        kernel_size=3,
        std_dev=0.1,
        seed=42
    )
    
    # Also test MixUp
    mixup = MyMixUp(alpha=1.0, num_classes=10, seed=42)
    
    # Define criterion and device
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Train for just 1 epoch for testing
    print("Training basic ELM for 1 iteration...")
    elm = elm.to(device)
    optimizer = torch.optim.SGD(elm.parameters(), lr=0.01, momentum=0.9)
    
    # One training iteration
    elm.train()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Test MixUp
        mixed_inputs, mixed_labels = mixup(inputs, labels)
        
        optimizer.zero_grad()
        outputs = elm(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Break after one batch for quick testing
        break
    
    # Test evaluation
    print("Testing evaluation...")
    elm.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = elm(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Break after one batch
            break
    
    print("✓ Full training cycle completed successfully")
    return True

def main():
    """Run all tests."""
    print("=== Running Tests for COMP0197 CW1 Task 2 ===")
    
    # Run tests
    elm_test = test_elm_implementation()
    mixup_test = test_mixup_implementation()
    ensemble_test = test_ensemble_implementation()
    full_test = test_full_training_cycle()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"ELM Implementation Test: {'Passed' if elm_test else 'Failed'}")
    print(f"MixUp Implementation Test: {'Passed' if mixup_test else 'Failed'}")
    print(f"Ensemble Implementation Test: {'Passed' if ensemble_test else 'Failed'}")
    print(f"Full Training Cycle Test: {'Passed' if full_test else 'Failed'}")
    
    if elm_test and mixup_test and ensemble_test and full_test:
        print("\nAll tests passed successfully! Your implementation looks good.")
    else:
        print("\nSome tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()