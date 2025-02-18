#%%
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
# Paths
train_dir = r"F:\Pranjal"  # Replace this with the path where images are stored
csv_path = os.path.join(train_dir, "defect_and_no_defect.csv")

# Load CSV
all_images_df = pd.read_csv(csv_path)

# Custom Dataset
class FaultDetectionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if available
        if self.transform:
            image = self.transform(image)

        # Create a pixel-wise mask filled with the image-level label
        mask = torch.ones((256, 256), dtype=torch.float32) * label

        # Add a channel dimension to the mask to make it (1, 256, 256)
        mask = mask.unsqueeze(0)

        return image, mask

# Define image transformations and create DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = FaultDetectionDataset(dataframe=all_images_df, img_dir=os.path.join(train_dir, "train_images"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# Checking the data loader
for images, labels in train_loader:
    print("Image batch shape:", images.size())
    print("Label batch shape:", labels.size())
    break
















# %%


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        
        # Example layers - replace these with your actual ResUNet layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # For binary classification (single output channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Instantiate the model
model = ResUNet()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



















# %%
# Training parameters
num_epochs = 5
train_losses = []

# Initialize the model and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResUNet().to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Suitable for segmentation tasks with binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)










#%%
import torch
from sklearn.metrics import classification_report
import numpy as np

def compute_classification_report(model, dataloader, device):
    model.eval()

    # Initialize lists to store true labels and predictions for calculating accuracy
    all_preds = []
    all_labels = []
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid to get probabilities (because of BCEWithLogitsLoss)
            preds = torch.sigmoid(outputs)  # Output probabilities between 0 and 1
            
            # Threshold the predictions (e.g., 0.5 for binary classification)
            preds = (preds > 0.5).float()

            # Calculate accuracy for the current batch
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.numel()

            # Flatten the predictions and labels for classification report
            all_preds.append(preds.view(-1).cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())
    
    # Convert lists to numpy arrays for classification report
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=["No Defect", "Defect"], zero_division=0)
    
    # Calculate accuracy
    accuracy = correct_preds / total_preds
    
    return report, accuracy

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Perform classification report and accuracy after every epoch, but only store them temporarily
    if (epoch + 1) % 1 == 0:  # e.g., every epoch
        try:
            report, accuracy = compute_classification_report(model, train_loader, device)
            print(f"Classification Report for Epoch {epoch+1}:\n{report}")
            print(f"Accuracy for Epoch {epoch+1}: {accuracy:.4f}")
        except MemoryError:
            print(f"Memory Error encountered at epoch {epoch + 1}. Skipping classification report calculation.")

print("Training complete!")

# Saving the model
model_save_path = "resunet_fault_detection.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
# %%
#%%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
# %%
