import torch
import torch.nn as nn
import torch.nn.functional as F

class MyExtremeLearningMachine(nn.Module):
    """
    Extreme Learning Machine implementation with:
    - One hidden convolutional layer with fixed (non-trainable) weights
    - One fully-connected layer with trainable weights
    """
    def __init__(self, input_channels=3, hidden_size=64, num_classes=10, kernel_size=5, std_dev=0.1, seed=42):
        """
        Initialize the ELM model.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB images)
            hidden_size (int): Number of feature maps in the hidden layer
            num_classes (int): Number of output classes
            kernel_size (int): Size of the convolutional kernel
            std_dev (float): Standard deviation for weight initialization
            seed (int): Random seed for reproducibility
        """
        super(MyExtremeLearningMachine, self).__init__()
        
        self.hidden_size = hidden_size
        self.std_dev = std_dev
        self.seed = seed
        
        # Define the convolutional layer with fixed weights
        self.conv = nn.Conv2d(input_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Calculate the output size after convolution (for CIFAR-10: 32x32)
        self.feature_size = 32 * 32 * hidden_size
        
        # Define the fully connected layer
        self.fc = nn.Linear(self.feature_size, num_classes)
        
        # Initialize fixed weights
        self.initialise_fixed_layers()
    
    def initialise_fixed_layers(self):
        """
        Initialize the fixed convolution kernel weights by random sampling 
        from a Gaussian distribution (mean = 0, with the given standard deviation).
        """
        # Set the random seed for reproducibility
        torch.manual_seed(self.seed)
        
        # Initialize the convolutional weights with Gaussian distribution
        nn.init.normal_(self.conv.weight, mean=0.0, std=self.std_dev)
        #nn.init.zeros_(self.conv.bias)
        
        # Freeze the convolutional layer weights
        self.conv.weight.requires_grad = False
        #self.conv.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # Pass through the fixed convolutional layer
        x = self.conv(x)
        
        # Apply non-linearity (ReLU)
        x = F.relu(x)
        
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Pass through the trainable fully connected layer
        x = self.fc(x)
        
        return x

class MyMixUp:
    """
    Implementation of MixUp data augmentation technique using torchvision.
    Includes probability parameter to control how often MixUp is applied.
    """
    def __init__(self, alpha=1.0, num_classes=10, prob=1.0, seed=42):
        """
        Initialize the MixUp augmentation.
        
        Args:
            alpha (float): Parameter for beta distribution
            num_classes (int): Number of classes in dataset
            prob (float): Probability of applying MixUp to a batch (0.0-1.0)
            seed (int): Random seed for reproducibility
        """
        self.alpha = alpha
        self.num_classes = num_classes
        self.prob = prob
        self.seed = seed
        
        # Generator for reproducibility
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        
        # Import the MixUp transform from torchvision
        try:
            from torchvision.transforms.v2 import MixUp as TorchMixUp
            self.mixup_transform = TorchMixUp(alpha=self.alpha, num_classes=self.num_classes)
        except ImportError:
            print("Could not import MixUp from torchvision.transforms.v2. Make sure you have the correct version.")
            # Fallback to custom implementation
            self.mixup_transform = None
    
    def __call__(self, x, y):
        """
        Apply MixUp augmentation to a batch of images and labels with probability self.prob.
        
        Args:
            x (torch.Tensor): Batch of images [batch_size, channels, height, width]
            y (torch.Tensor): Batch of labels [batch_size]
            
        Returns:
            tuple: (mixed_x, mixed_y) if MixUp is applied, otherwise (x, y_one_hot)
        """
        # Determine whether to apply MixUp based on probability
        apply_mixup = torch.rand(1, generator=self.generator).item() < self.prob
        
        # Convert labels to one-hot if they're not already
        if y.dim() == 1:
            y_one_hot = torch.zeros(y.size(0), self.num_classes, device=y.device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
        else:
            # Already one-hot
            y_one_hot = y
        
        if not apply_mixup:
            # Skip MixUp, return original data with one-hot labels
            return x, y_one_hot
        
        # Apply MixUp
        if self.mixup_transform is not None:
            # Apply the torchvision MixUp transform
            mixed_x, mixed_y = self.mixup_transform(x, y_one_hot)
            return mixed_x, mixed_y
        else:
            # Fallback to custom implementation if import failed
            batch_size = x.size(0)
            
            # Generate a random permutation of indices
            indices = torch.randperm(batch_size, generator=self.generator)
            
            # Get the mixed image and label pairs
            shuffled_x = x[indices]
            shuffled_y_one_hot = y_one_hot[indices]
            
            # Sample lambda from beta distribution
            if self.alpha > 0:
                lam = torch.distributions.Beta(self.alpha, self.alpha).sample(generator=self.generator)
            else:
                lam = 1.0
                
            # Mix images
            mixed_x = lam * x + (1 - lam) * shuffled_x
            
            # Mix labels
            mixed_y = lam * y_one_hot + (1 - lam) * shuffled_y_one_hot
            
            return mixed_x, mixed_y

class MyEnsembleELM(nn.Module):
    """
    Ensemble of multiple ELM models.
    """
    def __init__(self, input_channels=3, hidden_size=64, num_classes=10, kernel_size=5, 
                 std_dev=0.1, num_models=5, seed=42):
        """
        Initialize an ensemble of ELM models.
        
        Args:
            input_channels (int): Number of input channels
            hidden_size (int): Number of feature maps in each ELM
            num_classes (int): Number of output classes
            kernel_size (int): Size of the convolutional kernel
            std_dev (float): Standard deviation for weight initialization
            num_models (int): Number of ELM models in the ensemble
            seed (int): Random seed for reproducibility
        """
        super(MyEnsembleELM, self).__init__()
        
        # Parameter validation
        if hidden_size <= 0 or hidden_size > 512:
            print(f"Warning: hidden_size={hidden_size} is out of recommended range (1-512)")
        
        if std_dev <= 0 or std_dev > 1.0:
            print(f"Warning: std_dev={std_dev} is out of recommended range (0-1.0)")
        
        if num_models <= 0 or num_models > 20:
            print(f"Warning: num_models={num_models} is out of recommended range (1-20)")
        
        # Create a list of ELM models
        self.models = nn.ModuleList([
            MyExtremeLearningMachine(
                input_channels=input_channels,
                hidden_size=hidden_size,
                num_classes=num_classes,
                kernel_size=kernel_size,
                std_dev=std_dev,
                seed=seed + i  # Different seed for each model
            ) for i in range(num_models)
        ])
        
        self.num_models = num_models
    
    def forward(self, x):
        """
        Forward pass through all models in the ensemble.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Averaged output from all models
        """
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        
        # Stack outputs and average them
        outputs = torch.stack(outputs, dim=0)
        ensemble_output = torch.mean(outputs, dim=0)
        
        return ensemble_output