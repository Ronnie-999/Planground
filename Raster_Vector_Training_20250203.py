import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

# ---------------------------
# Data Augmentation and Transforms
# ---------------------------
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# ---------------------------
# Custom Dataset Class
# ---------------------------
class SingleInputDataset(Dataset):
    """
    Returns a tuple: (raster_tensor, coords_padded, wall_count)
      - raster_tensor: Tensor of shape [3, 256, 256] (RGB only)
      - coords_padded: Tensor of shape [max_walls * 4] (x1, y1, x2, y2 for each wall)
      - wall_count: int = number of actual walls
    """
    def __init__(self, raster_path, vector_path, max_walls=20):
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.max_walls = max_walls

        self.raster_files = sorted(os.listdir(raster_path))
        self.vector_files = sorted(os.listdir(vector_path))

        assert len(self.raster_files) == len(self.vector_files), \
            "Mismatch between number of raster files and vector JSON files!"

    def __len__(self):
        return len(self.raster_files)

    def __getitem__(self, idx):
        # ------------------------
        # 1) Load raster
        # ------------------------
        raster_file = self.raster_files[idx]
        raster_image = Image.open(os.path.join(self.raster_path, raster_file)).convert("RGB")
        original_width, original_height = raster_image.size

        # Apply data transforms
        raster_image = data_transforms(raster_image)  # shape [3, 256, 256]

        # ------------------------
        # 2) Load vector data (walls)
        # ------------------------
        vector_file = self.vector_files[idx]
        with open(os.path.join(self.vector_path, vector_file), "r") as f:
            vector_data = json.load(f)

        # The JSON structure: {
        #     "sample_id": 12,
        #     "walls": [
        #         {
        #             "id": 1,
        #             "type": "line",
        #             "x1": 242, "y1": 137,
        #             "x2": 179, "y2": 73
        #         },
        #         ...
        #     ]
        # }
        wall_list = vector_data.get("walls", [])
        wall_count = len(wall_list)

        # We'll store only up to max_walls
        # Each wall has 4 floats: x1, y1, x2, y2
        coords = []
        for i, wall in enumerate(wall_list[:self.max_walls]):
            # Normalize to 256x256
            x1 = wall["x1"] * 256.0 / original_width
            y1 = wall["y1"] * 256.0 / original_height
            x2 = wall["x2"] * 256.0 / original_width
            y2 = wall["y2"] * 256.0 / original_height
            coords.extend([x1, y1, x2, y2])

        # If there are fewer than max_walls, pad with zeros
        needed = self.max_walls * 4 - len(coords)
        coords_padded = coords + [0.0] * needed

        coords_padded = torch.tensor(coords_padded, dtype=torch.float32)
        wall_count = torch.tensor(wall_count, dtype=torch.long)
        
        return raster_image, coords_padded, wall_count

# ---------------------------
# Two-Headed Model
# ---------------------------
class UNetTwoHead(nn.Module):
    """
    1) Encodes an input [3 x 256 x 256] (RGB).
    2) Decodes to a feature map.
    3) Two heads:
       - self.count_head => classification for # of walls in [0..max_walls].
       - self.coord_head => coords for max_walls * 4 (x1,y1,x2,y2).
    """
    def __init__(self, input_channels=3, max_walls=20):
        super().__init__()
        self.max_walls = max_walls
        self.coords_dim = max_walls * 4  # total float coords per sample

        # Base backbone: MobileNet
        self.encoder = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Modify first conv layer to accept 'input_channels'
        self.encoder.features[0][0] = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.encoder.classifier = nn.Identity()

        # Simple upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # We'll flatten the decoder output, then feed into:
        #   1) count_head => # of walls (classification over [0..max_walls])
        #   2) coord_head => (max_walls*4) floats
        self.count_head = None
        self.coord_head = None

    def forward(self, x):
        # x shape => [B, 3, 256, 256]
        x = self.encoder.features(x)  # shape => [B, 1280, H/32, W/32]
        x = self.decoder(x)           # shape => [B, 64, ... , ...]
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)

        # Lazy init the heads so they know the exact in_features
        if self.count_head is None:
            self.count_head = nn.Linear(x_flat.size(1), self.max_walls + 1).to(x.device)
            self.coord_head = nn.Linear(x_flat.size(1), self.coords_dim).to(x.device)

        count_logits = self.count_head(x_flat)    # shape => [B, max_walls+1]
        coords = self.coord_head(x_flat)          # shape => [B, coords_dim]
        return count_logits, coords

# ---------------------------
# Data Paths
# ---------------------------
raster_path = r"C:\Users\MBodrov\Playground_dataset\version6\raster"
vector_path = r"C:\Users\MBodrov\Playground_dataset\version6\vector"
checkpoint_path = r"C:\Users\MBodrov\Playground_dataset\best_model_twohead_v6.pth"

# ---------------------------
# Hyperparams
# ---------------------------
max_walls = 20
batch_size = 4
learning_rate = 1e-3
num_epochs = 50
patience = 5
train_val_split = 0.8

# ---------------------------
# Dataset & Dataloader
# ---------------------------
dataset = SingleInputDataset(raster_path, vector_path, max_walls=max_walls)
train_size = int(train_val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

def custom_collate_fn(batch):
    # batch is list of (input, coords_padded, wall_count)
    Xs, coords, counts = zip(*batch)
    Xs = torch.stack(Xs, dim=0)           # shape => [B, 3, 256, 256]
    coords = torch.stack(coords, dim=0)   # shape => [B, max_walls*4]
    counts = torch.stack(counts, dim=0)   # shape => [B]
    return Xs, coords, counts

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=custom_collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        collate_fn=custom_collate_fn, num_workers=0)

# ---------------------------
# Define Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetTwoHead(input_channels=3, max_walls=max_walls).to(device)

# ---------------------------
# Define Losses
# ---------------------------
count_loss_fn = nn.CrossEntropyLoss()

def masked_mse_coords(pred_coords, true_coords, true_counts):
    """
    pred_coords, true_coords: shape [B, max_walls * 4]
    true_counts: shape [B]  integer wall count
    """
    B, total_dim = pred_coords.shape
    max_floats = 4 * max_walls  # (x1,y1,x2,y2) per wall

    # Build a mask that is 1.0 for valid wall coords, 0.0 for the padded part
    mask = torch.zeros_like(pred_coords)
    for i in range(B):
        count_i = true_counts[i].item()
        valid_len = min(count_i * 4, max_floats)
        mask[i, :valid_len] = 1.0

    diff = (pred_coords - true_coords) * mask
    diff_sq = diff ** 2
    valid_count = mask.sum() + 1e-8
    mse = diff_sq.sum() / valid_count
    return mse

def total_loss_fn(count_logits, coords_pred, true_count, coords_true):
    # 1) classification for number of walls
    loss_count = count_loss_fn(count_logits, true_count)
    # 2) masked MSE for wall coordinates
    loss_coords = masked_mse_coords(coords_pred, coords_true, true_count)
    return loss_count + loss_coords

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ---------------------------
# Training Loop
# ---------------------------
def plot_progress(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Progress (Two-Headed Model)")
    plt.show()

train_losses = []
val_losses = []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for X_batch, coords_batch, count_batch in tqdm(train_loader, desc="Train", leave=False):
        X_batch = X_batch.to(device)
        coords_batch = coords_batch.to(device)
        count_batch = count_batch.to(device)

        optimizer.zero_grad()
        count_logits, coords_pred = model(X_batch)
        loss = total_loss_fn(count_logits, coords_pred, count_batch, coords_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for X_batch, coords_batch, count_batch in tqdm(val_loader, desc="Val", leave=False):
            X_batch = X_batch.to(device)
            coords_batch = coords_batch.to(device)
            count_batch = count_batch.to(device)

            count_logits, coords_pred = model(X_batch)
            loss = total_loss_fn(count_logits, coords_pred, count_batch, coords_batch)
            val_running_loss += loss.item()

    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1} -- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Step scheduler
    scheduler.step()

    # Early stopping / checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"** Saved best model at epoch {epoch+1}.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Plot progress
    plot_progress(train_losses, val_losses)

# Final Plot
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Final Training Curves")
plt.show()

print("Training complete.")

