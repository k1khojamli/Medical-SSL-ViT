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


class MRIDataset(Dataset):
    def __init__(self,data_path,labels,augment=True):
        super().__init__()
        self.data = np.load(data_path, mmap_mode='r')
        self.labels = np.load(labels)
        self.augment=augment
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_org = self.data[index]
        label = self.labels[index]
        
        if (random.random() < 0.5 ) & self.augment:
            augmented = transform(image=image_org)
            image = augmented["image"]
        else:
            image = image_org
        
        edges = cv2.Canny(image.astype(np.uint8), 30, 100, apertureSize=5, L2gradient=True)
        
        
        # Stack all channels (augmented image, edges)
        multi_channel_image = np.stack([image, edges], axis=-1)  # Shape (256, 256, 2)

        return (
            torch.tensor(multi_channel_image, dtype=torch.float16).permute(2, 0, 1)/255.0,
            torch.tensor(label, dtype=torch.float16)
        )


# Define the augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=10, p=0.6),  # Rotate within ±30 degrees
    A.GaussNoise(std_range=(0.01,0.1), p=0.5),  # Add Gaussian noise
    A.RandomBrightnessContrast(p=0.7),  # Simulate artifacts with brightness/contrast changes
    A.Affine(translate_percent=(-0.1, 0.1), scale=(0.9, 1.1), p=0.7),  # Linear movement and zooming
])

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=256, img_size=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Compute number of patches
        num_patches = (img_size // patch_size) ** 2

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, emb_dim))
    
    def forward(self, x):
        x = self.proj(x)  # Shape: (B, emb_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten patches

        x = x + self.pos_embed
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, img_size=256, in_channels=1, patch_size=16,latent_space =256, emb_dim=256, depth=6, num_heads=8, mlp_dim=512):
        super().__init__()
                
        self.patch_embed = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, emb_dim=emb_dim, img_size=img_size)
        self.transformer = nn.Sequential(*[TransformerEncoder(emb_dim, num_heads, mlp_dim) for _ in range(depth)])
        self.linear_proj = nn.Linear((img_size // patch_size) ** 2 * emb_dim, latent_space)  # Flatten into latent space

    
    def forward(self, x):      
        x = self.patch_embed(x)
        x = self.transformer(x)
        
        x = x.contiguous().view(x.shape[0], -1)
        x = self.linear_proj(x)

        return x
    

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, latent_dim=384, hidden_dim=128, num_classes=1):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out
    


train_data = MRIDataset(
    data_path="../../data_labeled_pretrain_HM/train.npy",
    labels="../../laststep/data/train_label.npy",
    augment=True
)
test_data = MRIDataset(
    data_path="../../data_labeled_pretrain_HM/test.npy",
    labels="../../laststep/data/test_label.npy",
    augment=False
)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = DataLoader(test_data, batch_size=310, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(img_size=256, in_channels=2, patch_size=16,
                  latent_space=384, emb_dim=512, depth=6, num_heads=8, mlp_dim=512).to(device)

classifier = EncoderClassifier(
    encoder=encoder,
    latent_dim=384,
    hidden_dim=256,
    num_classes=1
).to(device)
scaler = GradScaler(enabled=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

epochs = 50
train_losses, test_losses = [], []

for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    for batch_idx,(x, y) in tqdm(enumerate(train_loader),total=len(train_loader)):
        x, y = x.to(device), y.to(device)

        with autocast(device_type='cuda'):
            logits = classifier(x).view(-1)  
            loss = criterion(logits, y)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    #evaluation
    classifier.eval()
    test_loss =0
    with torch.no_grad():
        for batch_idx,(x, y) in tqdm(enumerate(test_loader),total=len(test_loader)):
            x, y = x.to(device), y.to(device)
            with autocast(device_type='cuda'):
                logits = classifier(x).view(-1) 
                loss = criterion(logits, y)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Save model after 100 epochs
    if (epoch+1) % 50 == 0:
        torch.save(classifier.state_dict(), f"classifier_epoch_{epoch+1}.pth")
