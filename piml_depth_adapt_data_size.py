import os
import cv2
import pywt
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
from thop import profile
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from depth_model_zoo import SimpleCNN, build_depth_regression_model


robot_name = 'robot_1_eight_ball'
# 'robot_2_four_ball' 'robot_3_stable_bot'
data_root = f"./Image_Depth/{robot_name}"
os.makedirs(f"./checkpoint_data_size/{robot_name}", exist_ok=True)
txt_file = [f for f in os.listdir(data_root) if f.endswith(".txt")][0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# size = 16
g_s =(4, 4)
fine_grid_size=(6, 6)
coarse_grid_size=(4, 4)
piml = False #False # True
data_sizes = [0.8, 0.6, 0.4, 0.2]

total_fine = fine_grid_size[0] * fine_grid_size[1]
total_coarse = coarse_grid_size[0] * coarse_grid_size[1]
total_needed = total_fine + total_coarse

# === 1. Focus Measure Functions ===
def laplacian_of_gaussian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def tenengrad(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(gradient_magnitude)

def brenner(image):
    shifted_image = np.roll(image, -2, axis=1)
    return np.sum((image[:, :-2] - shifted_image[:, :-2])**2)

def gray_level_variance(image):
    return np.var(image)

def entropy(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram / histogram.sum()
    entropy_value = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy_value

def max_absolute_gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.max(np.abs(sobel_x) + np.abs(sobel_y))

focus_functions = [
    laplacian_of_gaussian,
    tenengrad, brenner,
    gray_level_variance,
    entropy, 
    max_absolute_gradient
]
print(f"Robot: {robot_name}, PIML: {piml}, Grid Size: {fine_grid_size, coarse_grid_size}, focus feature dim {len(focus_functions)*g_s[0]*g_s[1]}")


# === 2. Segment image ===
def segment_image(image, grid_size):
    segments = []
    h, w = image.shape
    step_h, step_w = h // grid_size[0], w // grid_size[1]
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            segment = image[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w]
            segments.append(segment)
    return segments

def adaptive_segment_image(image, fine_grid=(6, 6), coarse_grid=(4, 4), fine_region_expand_ratio=1):
    h, w = image.shape
    segments = []

    # Step 1: Find object region using contour
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # fallback: divide whole image as coarse grid
        print("No contours found. Using uniform coarse grid.")
        return segment_image(image, coarse_grid)

    # Step 2: Compute bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Optionally expand the bounding box
    cx, cy = x + bw // 2, y + bh // 2
    bw = int(bw * fine_region_expand_ratio)
    bh = int(bh * fine_region_expand_ratio)
    x = max(cx - bw // 2, 0)
    y = max(cy - bh // 2, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    # Step 3: Crop and divide the object region using fine grid
    object_region = image[y:y+bh, x:x+bw]
    fine_h, fine_w = bh // fine_grid[0], bw // fine_grid[1]
    for i in range(fine_grid[0]):
        for j in range(fine_grid[1]):
            seg = object_region[i*fine_h:(i+1)*fine_h, j*fine_w:(j+1)*fine_w]
            segments.append(seg)
    # print(f"Fine Segment : {len(segments)} Girds")
    
    # Step 4: Divide the background (excluding object region) using coarse grid
    mask = np.zeros_like(image, dtype=bool)
    mask[y:y+bh, x:x+bw] = True

    coarse_h, coarse_w = h // coarse_grid[0], w // coarse_grid[1]
    for i in range(coarse_grid[0]):
        for j in range(coarse_grid[1]):
            top, left = i * coarse_h, j * coarse_w
            patch = image[top:top+coarse_h, left:left+coarse_w]
            if not mask[top:top+coarse_h, left:left+coarse_w].any():
                segments.append(patch)
                
    # print(f"Totall: {len(segments)} Girds")
    # Pad fine segments if not enough (rare case)
    while len(segments) < total_needed:
        zero_patch = np.zeros_like(segments[0])
        segments.append(zero_patch)

    # print(f"✅ After extend, we have to {len(segments)} Girds")

    return segments

# === 3. Feature Extraction ===
def extract_local_features(image, grid_size, focus_functions):
    segments = adaptive_segment_image(image, fine_grid=fine_grid_size, coarse_grid=coarse_grid_size)
    features = []
    for segment in segments:
        features.extend([func(segment) for func in focus_functions])
    return features

def process_tensor_image(tensor_img, grid_size=(4, 4)):
    assert tensor_img.shape == (1, 3, 224, 224), "Expected shape (1, 3, 224, 224)"
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (224, 224, 3)
    img_gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    features = extract_local_features(img_gray, grid_size, focus_functions)
    return features

# Dataset
class DepthDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)

        if piml == True:
            image_tensor = image.unsqueeze(0)  # add batch dim for focus measure
            focus_feat = process_tensor_image(image_tensor, grid_size=g_s)
            focus_feat_tensor = torch.tensor(focus_feat, dtype=torch.float32)
        else:
            focus_feat_tensor = torch.randn(1, len(focus_functions)*total_needed).to(device)

        return image, focus_feat_tensor, torch.tensor(row['depth'], dtype=torch.float32)

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


class PhysicsInformedResNet50(nn.Module):
    def __init__(self, physics_feat_dim=len(focus_functions)*total_needed, physics_hidden_dim=1): #len(focus_functions) * g_s[0]*g_s[1]
        super(PhysicsInformedResNet50, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 2048, 1, 1]
        self.img_feat_dim = 2048

        # Physics feature encoder
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_feat_dim, physics_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(physics_hidden_dim))

        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.img_feat_dim + physics_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, x_img, x_physics):
        # Image feature extraction
        x_img = self.backbone(x_img)           # [B, 2048, 1, 1]
        x_img = torch.flatten(x_img, 1)        # [B, 2048]

        # Physics feature encoding
        x_physics = self.physics_encoder(x_physics)  # [B, physics_hidden_dim]

        # Concatenate and regress
        x = torch.cat([x_img, x_physics], dim=1)     # [B, 2048 + hidden]
        out = self.regressor(x)                      # [B, 1]
        return out.squeeze(1)
        
class ResNet50Baseline(nn.Module):
    def __init__(self):
        super(ResNet50Baseline, self).__init__()
        resnet = models.resnet50(pretrained=True)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 1)
        self.model = resnet

    def forward(self, x, physics_feat):
        return self.model(x).squeeze(1)  # [B]

def train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix, epochs=30):
    best_mse = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, focus_feat_tensor, targets in train_loader:
            imgs, focus_feat_tensor, targets = imgs.to(device), focus_feat_tensor.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs, focus_feat_tensor).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss = 0.0
            
        print(f"Epoch {epoch+1} - Total Loss: {total_loss*1000:.4f}")
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
        for imgs, focus_feat_tensor, targets in loader:
            imgs, focus_feat_tensor = imgs.to(device), focus_feat_tensor.to(device)
            preds = model(imgs, focus_feat_tensor).squeeze().cpu().numpy()
            # print(preds.shape)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    return mean_squared_error(all_targets, all_preds)

def test_model(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, focus_feat_tensor, targets in loader:
            imgs, focus_feat_tensor = imgs.to(device), focus_feat_tensor.to(device)
            preds = model(imgs, focus_feat_tensor).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return mse, r2

# Main loop
results = []
# print(f"\n🚀 Training RESNET50...Grid Size {g_s[0],g_s[1]}")

for size in data_sizes:
    print(f"\n🚀 Training PIML-{piml}... with data {size*100}%")
    # dfs = df.sample(frac=size, random_state=42).reset_index(drop=True)
    # Split data
    train_df_s, temp_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df= train_df_s.sample(frac=size, random_state=42).reset_index(drop=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_ds = DepthDataset(train_df)
    val_ds = DepthDataset(val_df)
    test_ds = DepthDataset(test_df)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=16, drop_last=True)

    if piml == True:
        model = PhysicsInformedResNet50().to(device)
    else:
        model = ResNet50Baseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    save_prefix = f"checkpoint_data_size/{robot_name}/{robot_name}_{size}_grid-{piml}"
    
    train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix)
    # model.load_state_dict(torch.load(f"{save_prefix}_latest.pth"))
    model.load_state_dict(torch.load(f"{save_prefix}_best.pth"))
    mse, r2 = test_model(model, test_loader)

# input_tensor = torch.randn(1, 3, 224, 224).to(device)
# focus_tensor_test = torch.randn(1, len(focus_functions)*total_needed).to(device)
# macs, params = profile(model, inputs=(input_tensor,focus_tensor_test), verbose=False)

    results.append({
        "Grids": piml,
        "Size": size,
        "MSE": mse,
        "R2": r2
    })
    print(f"✅ PIML-{piml}, Size-{size}, MSE: {mse:.4f}, R^2: {r2:.4f}")

# Save results
output_csv = f"./checkpoint_data_size/{robot_name}/results_grid-{piml}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 Results saved to {output_csv}")