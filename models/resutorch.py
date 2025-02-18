#%%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#%%
torch.cuda.empty_cache()

#%%
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()  # Initialize a gradient scaler for mixed precision training
# Data preprocessing: adjust the image size
from torchvision import transforms


# %%
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


# %%
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths and Data Preparation
train_dir = r"C:\Pranjal"  # Update with your directory path where images and CSV are located
all_images_df = pd.read_csv(os.path.join(train_dir, "defect_and_no_defect.csv"))

# Split data
train_df, test_df = train_test_split(all_images_df, test_size=0.15, random_state=42)

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.dataframe['label'] = self.label_encoder.fit_transform(self.dataframe['label'].astype(str))
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transformations and Dataloaders
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce input image size to 128x128
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(train_df, os.path.join(train_dir, 'train_images'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

test_dataset = CustomDataset(test_df, os.path.join(train_dir, 'train_images'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# %%
# Simplified Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        return torch.relu(x)

# Simplified ResUNet Model
class ExtraLightResUNet(nn.Module):
    def __init__(self):
        super(ExtraLightResUNet, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU())
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, padding=1), nn.ReLU())
        
        # Output layer
        self.conv_out = nn.Conv2d(8, 2, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        
        bridge = self.bridge(self.pool2(enc2))
        
        dec2 = self.dec2(torch.cat([self.upconv2(bridge), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        return self.conv_out(dec1)

model = ExtraLightResUNet().to(device)


# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%
# Training Loop
num_epochs = 5
accumulation_steps = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(train_loader):
        # Clear GPU cache
        torch.cuda.empty_cache()
        if labels.dim() > 1:  # Check if there is more than one dimension
            labels = labels.squeeze(1)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        with autocast():  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss
            
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights after accumulating gradients
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}")
# Save the simplified model
torch.save(model.state_dict(), "simplified_resunet_model.pth")
print("Model saved as 'simplified_resunet_model.pth'.")

# %%
