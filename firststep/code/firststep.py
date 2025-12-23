import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from einops import rearrange
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random
import cv2
import matplotlib.pyplot as plt


# =========================
# Custom Dataset for MRI Data
# =========================

class MRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing MRI images.
    Supports data augmentation and multi-channel input (original + edge detection).
    """
    def __init__(self, data_path, labels, augment=True):
        """
        Initialize MRI Dataset.
        
        Args:
            data_path (str): Path to numpy file containing MRI images
            labels (str): Path to numpy file containing labels
            augment (bool): Whether to apply data augmentation (default: True)
        """
        super().__init__()
        # Load data with memory mapping to reduce RAM usage
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels)
        self.augment = augment
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (multi_channel_image, label)
                - multi_channel_image: Tensor of shape (2, 256, 256) containing original and edge channels
                - label: Tensor containing the binary label
        """
        # Load image and corresponding label
        image_org = self.data[index]
        label = self.labels[index]
        
        # Apply augmentation with 50% probability during training
        if (random.random() < 0.5) and self.augment:
            augmented = transform(image=image_org)
            image = augmented["image"]
        else:
            image = image_org
        
        # Extract edges using Canny edge detection
        # apertureSize=5: Size of Sobel kernel for gradient calculation
        # L2gradient=True: Use L2 norm for more accurate gradient magnitude
        edges = cv2.Canny(image.astype(np.uint8), 30, 100, apertureSize=5, L2gradient=True)
        
        # Stack original image and edge map as separate channels
        # Shape: (256, 256, 2)
        multi_channel_image = np.stack([image, edges], axis=-1)

        # Convert to PyTorch tensor, permute to (C, H, W) format, and normalize to [0, 1]
        return (
            torch.tensor(multi_channel_image, dtype=torch.float16).permute(2, 0, 1) / 255.0,
            torch.tensor(label, dtype=torch.float16)
        )


# =========================
# Data Augmentation Pipeline
# =========================

# Define augmentation transforms using Albumentations library
transform = A.Compose([
    A.Rotate(limit=10, p=0.6),  # Random rotation within ±10 degrees with 60% probability
    A.GaussNoise(std_range=(0.01, 0.1), p=0.5),  # Add Gaussian noise to simulate scanner artifacts
    A.RandomBrightnessContrast(p=0.7),  # Random brightness and contrast adjustments
    A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), p=0.7),  # Translation and scaling transforms
])


# =========================
# Vision Transformer Components
# =========================

class PatchEmbedding(nn.Module):
    """
    Convert input image into patch embeddings with positional encoding.
    Splits the image into non-overlapping patches and projects them to embedding dimension.
    """
    def __init__(self, in_channels=1, patch_size=16, emb_dim=256, img_size=256):
        """
        Initialize Patch Embedding layer.
        
        Args:
            in_channels (int): Number of input channels (default: 1)
            patch_size (int): Size of each patch (default: 16)
            emb_dim (int): Embedding dimension (default: 256)
            img_size (int): Input image size (default: 256)
        """
        super().__init__()
        # Convolutional layer to split image into patches and project to embedding space
        # kernel_size = stride = patch_size ensures non-overlapping patches
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Calculate total number of patches
        # For 256x256 image with 16x16 patches: (256/16)^2 = 256 patches
        num_patches = (img_size // patch_size) ** 2

        # Learnable positional embeddings to encode spatial information
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))
    
    def forward(self, x):
        """
        Forward pass: Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor of shape (B, num_patches, emb_dim) with positional embeddings added
        """
        # Project and split into patches: (B, emb_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, emb_dim, H/P, W/P) -> (B, num_patches, emb_dim)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add positional embeddings to retain spatial information
        x = x + self.pos_embed
        
        return x


class TransformerEncoder(nn.Module):
    """
    Single Transformer Encoder block with Multi-Head Self-Attention and MLP.
    Implements the standard Transformer architecture with pre-normalization.
    """
    def __init__(self, emb_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        """
        Initialize Transformer Encoder block.
        
        Args:
            emb_dim (int): Embedding dimension (default: 256)
            num_heads (int): Number of attention heads (default: 8)
            mlp_dim (int): Hidden dimension of MLP (default: 512)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()
        # Layer normalization before attention
        self.norm1 = nn.LayerNorm(emb_dim)
        
        # Multi-Head Self-Attention mechanism
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(emb_dim)
        
        # Feed-forward MLP with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),  # Gaussian Error Linear Unit - smoother than ReLU
            nn.Linear(mlp_dim, emb_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor of shape (B, num_patches, emb_dim)
            
        Returns:
            Output tensor of same shape with attention and MLP applied
        """
        # Self-attention with residual connection
        # Pre-normalization: norm is applied before attention
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        
        # MLP with residual connection
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x


class Encoder(nn.Module):
    """
    Vision Transformer Encoder that processes images as sequences of patches.
    Consists of patch embedding followed by multiple Transformer encoder layers.
    """
    def __init__(self, img_size=256, in_channels=1, patch_size=16, latent_space=256, 
                 emb_dim=256, depth=6, num_heads=8, mlp_dim=512):
        """
        Initialize Vision Transformer Encoder.
        
        Args:
            img_size (int): Input image size (default: 256)
            in_channels (int): Number of input channels (default: 1)
            patch_size (int): Size of each patch (default: 16)
            latent_space (int): Dimension of output latent space (default: 256)
            emb_dim (int): Embedding dimension (default: 256)
            depth (int): Number of Transformer encoder blocks (default: 6)
            num_heads (int): Number of attention heads (default: 8)
            mlp_dim (int): Hidden dimension of MLP in Transformer (default: 512)
        """
        super().__init__()
        
        # Patch embedding layer to convert image to sequence of embeddings        
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, 
                                         emb_dim=emb_dim, img_size=img_size)
        
        # Stack of Transformer encoder blocks
        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        
        # Linear projection to compress all patch embeddings into latent space
        # Input: flattened patches (num_patches * emb_dim)
        # Output: latent vector (latent_space)
        self.linear_proj = nn.Linear((img_size // patch_size) ** 2 * emb_dim, latent_space)

    def forward(self, x):
        """
        Forward pass: Image -> Patches -> Transformer -> Latent representation.
        
        Args:
            x: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Latent representation of shape (B, latent_space)
        """
        # Convert image to patch embeddings: (B, num_patches, emb_dim)
        x = self.patch_embed(x)
        
        # Process through Transformer encoder blocks
        x = self.transformer(x)
        
        # Flatten all patch embeddings: (B, num_patches, emb_dim) -> (B, num_patches * emb_dim)
        x = x.contiguous().view(x.shape[0], -1)
        
        # Project to latent space: (B, num_patches * emb_dim) -> (B, latent_space)
        x = self.linear_proj(x)

        return x
    

class EncoderClassifier(nn.Module):
    """
    Complete classification model combining Vision Transformer encoder with MLP classifier head.
    Used for binary classification of MRI images.
    """
    def __init__(self, encoder, latent_dim=384, hidden_dim=128, num_classes=1):
        """
        Initialize the classifier model.
        
        Args:
            encoder: Pre-defined encoder (Vision Transformer)
            latent_dim (int): Dimension of encoder's latent space (default: 384)
            hidden_dim (int): Hidden dimension of classifier MLP (default: 128)
            num_classes (int): Number of output classes (1 for binary classification)
        """
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        
        # Classification head: MLP with dropout for regularization
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass: Extract features and classify.
        
        Args:
            x: Input image tensor
            
        Returns:
            Classification logits (use with BCEWithLogitsLoss)
        """
        # Extract features using encoder
        features = self.encoder(x)
        
        # Classify using MLP head
        out = self.fc(features)
        
        return out


# =========================
# Dataset Loading
# =========================

# Initialize training dataset with augmentation enabled
train_data = MRIDataset(
    data_path="../../data_labeled_pretrain_HM/train.npy",
    labels="../../laststep/data/train_label.npy",
    augment=True
)

# Initialize test dataset without augmentation
test_data = MRIDataset(
    data_path="../../data_labeled_pretrain_HM/test.npy",
    labels="../../laststep/data/test_label.npy",
    augment=False
)

# Create data loaders for batch processing
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=310, shuffle=False)


# =========================
# Model Initialization
# =========================

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Vision Transformer encoder
# in_channels=2: original image + edge map
encoder = Encoder(
    img_size=256,           # Input image size
    in_channels=2,          # Two channels: original + edges
    patch_size=16,          # 16x16 patches
    latent_space=384,       # Latent representation dimension
    emb_dim=512,            # Embedding dimension
    depth=6,                # 6 Transformer encoder blocks
    num_heads=8,            # 8 attention heads
    mlp_dim=512             # MLP hidden dimension
).to(device)

# Initialize classifier with encoder
classifier = EncoderClassifier(
    encoder=encoder,
    latent_dim=384,         # Must match encoder's latent_space
    hidden_dim=256,         # Classifier MLP hidden dimension
    num_classes=1           # Binary classification
).to(device)


# =========================
# Training Setup
# =========================

# Gradient scaler for automatic mixed precision training (AMP)
# Helps reduce memory usage and speed up training
scaler = GradScaler(enabled=True)

# Binary Cross-Entropy loss with logits (includes sigmoid)
criterion = nn.BCEWithLogitsLoss()

# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

# Training configuration
epochs = 50
train_losses, test_losses = [], []  # Track losses for visualization


# =========================
# Training Loop
# =========================

for epoch in range(epochs):
    # ========== Training Phase ==========
    classifier.train()  # Set model to training mode (enables dropout, batch norm, etc.)
    total_loss = 0
    
    # Iterate through training batches with progress bar
    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # Move data to GPU
        x, y = x.to(device), y.to(device)

        # Mixed precision training context
        with autocast(device_type='cuda'):
            # Forward pass: get model predictions
            logits = classifier(x).view(-1)  # Flatten to 1D for binary classification
            
            # Calculate loss
            loss = criterion(logits, y)
        
        # Backpropagation with gradient scaling
        optimizer.zero_grad()           # Clear previous gradients
        scaler.scale(loss).backward()   # Backward pass with scaled gradients
        scaler.step(optimizer)          # Update weights with unscaled gradients
        scaler.update()                 # Update gradient scaler
        
        # Accumulate loss for epoch average
        total_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_train_loss = total_loss / len(train_loader)

    # ========== Evaluation Phase ==========
    classifier.eval()  # Set model to evaluation mode (disables dropout, etc.)
    test_loss = 0
    
    # Disable gradient computation for evaluation (saves memory and computation)
    with torch.no_grad():
        for batch_idx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            # Move data to GPU
            x, y = x.to(device), y.to(device)
            
            # Mixed precision evaluation
            with autocast(device_type='cuda'):
                logits = classifier(x).view(-1)
                loss = criterion(logits, y)
            
            # Accumulate test loss
            test_loss += loss.item()
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)

    # Store losses for plotting
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Save model checkpoint every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(classifier.state_dict(), f"classifier_epoch_{epoch+1}.pth")
        print(f"Model saved: classifier_epoch_{epoch+1}.pth")

# =========================
# Visualization - Training and Test Loss
# =========================

# Plot loss curves to monitor training progress
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Test Loss")
plt.show()


# =========================
# Import Required Metrics
# =========================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.amp import autocast


# =========================
# Function: Evaluate Classification Metrics with AMP
# =========================

def evaluate_metrics_amp(model, loader, device, threshold=0.5):
    """
    Evaluate model performance using standard classification metrics.
    
    Args:
        model: PyTorch model to evaluate
        loader: DataLoader containing validation/test data
        device: Device to run evaluation on ('cuda' or 'cpu')
        threshold: Probability threshold for binary classification (default: 0.5)
    
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for x, y in loader:
            # Move batch to device
            x, y = x.to(device), y.to(device)
            
            # Use automatic mixed precision for faster inference
            with autocast('cuda'):
                # Get model predictions (logits)
                logits = model(x).view(-1)
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                
                # Apply threshold to get binary predictions
                preds = (probs > threshold).float()
            
            # Store predictions and labels on CPU
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    # Concatenate all batches and convert to numpy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Calculate classification metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return acc, prec, rec, f1


# =========================
# Evaluate Metrics on Training Set
# =========================

train_acc, train_prec, train_rec, train_f1 = evaluate_metrics_amp(classifier, train_loader, device)
print(f"Train Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")


# =========================
# Evaluate Metrics on Test Set
# =========================

test_acc, test_prec, test_rec, test_f1 = evaluate_metrics_amp(classifier, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")


# =========================
# Import Confusion Matrix
# =========================

from sklearn.metrics import confusion_matrix
from torch.amp import autocast


# =========================
# Function: Compute Detailed Confusion Matrix
# =========================

def confusion_counts_verbose(model, loader, device, threshold=0.5):
    """
    Compute detailed confusion counts for a binary classification model.
    
    Arguments:
        model: PyTorch model
        loader: DataLoader containing (inputs, labels)
        device: 'cuda' or 'cpu'
        threshold: threshold for converting probabilities to binary predictions
    
    Returns:
        tn, fp, fn, tp: counts of True Negatives, False Positives, False Negatives, True Positives
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to accumulate predictions and labels
    all_preds = []
    all_labels = []
    
    # Disable gradient computation
    with torch.no_grad():
        for x, y in loader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            # AMP for mixed precision
            with autocast('cuda'):
                # Get logits from model - shape: (batch_size,)
                logits = model(x).view(-1)
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                
                # Apply threshold to get 0/1 predictions
                preds = (probs > threshold).float()
            
            # Store on CPU
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    
    # Concatenate all batches and convert to numpy
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Compute confusion matrix: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Print detailed confusion matrix breakdown
    print("=== Confusion Counts ===")
    print(f"True Positives (TP): {tp} -> Number of positive samples correctly predicted as 1")
    print(f"False Negatives (FN): {fn} -> Number of positive samples incorrectly predicted as 0")
    print(f"True Negatives (TN): {tn} -> Number of negative samples correctly predicted as 0")
    print(f"False Positives (FP): {fp} -> Number of negative samples incorrectly predicted as 1")
    print("========================")
    
    return tn, fp, fn, tp


# =========================
# Compute Confusion Matrix for Training Set
# =========================

print("Train Confusion Counts:")
confusion_counts_verbose(classifier, train_loader, device)


# =========================
# Compute Confusion Matrix for Test Set
# =========================

print("\nTest Confusion Counts:")
confusion_counts_verbose(classifier, test_loader, device)

