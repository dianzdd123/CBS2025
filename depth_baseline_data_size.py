# Data Size Experiment
import os
import random
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import ViTModel
from thop import profile
from depth_model_zoo import SimpleCNN, build_depth_regression_model

# Settings
robot_name = "robot_8_robot"
data_root = f"./Image_Depth/{robot_name}"
txt_file = [f for f in os.listdir(data_root) if f.endswith(".txt")][0]
# data_sizes = [1.0, 0.8, 0.6, 0.4, 0.2]
data_sizes = [0.8, 0.6, 0.4, 0.2]
model_names = ["resnet50"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(f"./checkpoint_Depth_Datasize/{robot_name}", exist_ok=True)
output_csv = f"./checkpoint_Depth_Datasize/{robot_name}/depth_regression_results_{robot_name}.csv"

# Dataset
class DepthDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(row['depth'], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

# Load image paths and depth
depths = []
with open(os.path.join(data_root, txt_file), 'r') as f:
    for line in f:
        parts = line.strip().split()
        depth = float(parts[-1])
        depths.append(depth)

image_files = sorted([f for f in os.listdir(data_root) if f.lower().endswith(('jpg', 'png'))])
image_paths = [os.path.join(data_root, f) for f in image_files]
df = pd.DataFrame({"image_path": image_paths, "depth": depths})

def train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix, epochs=10):
    best_mse = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_samples = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs).reshape(-1)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            seen_samples += imgs.size(0)
            
        if not seen_samples:
            raise ValueError("Training loader produced no batches; check dataset size and batch size.")
        print(f"Epoch {epoch+1} - Mean training loss: {total_loss / seen_samples:.4f}")
        torch.save(model.state_dict(), f"{save_prefix}_latest.pth")

        val_mse = evaluate(model, val_loader)
        print(f"          MSE: {val_mse}")
        if val_mse < best_mse:
            best_mse = val_mse
            torch.save(model.state_dict(), f"{save_prefix}_best.pth")
    return best_mse

def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs).reshape(-1).cpu().numpy()
            # print(preds.shape)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return mean_squared_error(all_targets, all_preds)

def test_model(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs).reshape(-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return mse, r2

# Main loop
results = []

for size in data_sizes:
    dfs = df.sample(frac=size, random_state=42).reset_index(drop=True)
    # Split data
    train_df, temp_df = train_test_split(dfs, test_size=0.1, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    train_ds = DepthDataset(train_df)
    val_ds = DepthDataset(val_df)
    test_ds = DepthDataset(test_df)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=16, drop_last=False)
    
    for model_name in model_names:
        print(f"\n🚀 Training {model_name.upper()}... with data {size*100}%")
        model = build_depth_regression_model(model_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if model_name in ["vgg16", "vit"] else 1e-3)
        criterion = nn.MSELoss()
        save_prefix = f"checkpoint_Depth_Datasize/{robot_name}/{robot_name}_{model_name}_{size}"
        
        train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix)
        model.load_state_dict(torch.load(f"{save_prefix}_best.pth"))
        mse, r2 = test_model(model, test_loader)
    
        # input_tensor = torch.randn(1, 3, 224, 224).to(device)
        # macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    
        results.append({
            "Model": model_name,
            "Size": size,
            "MSE": mse,
            "R2": r2
        })
        print(f"✅ {model_name}-{size} done. MSE: {mse:.4f}, R^2: {r2:.4f}")

# Save results
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 Results saved to {output_csv}")