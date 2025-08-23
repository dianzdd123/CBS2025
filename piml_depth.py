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


robot_name = 'robot_3_stable_bot'
# 'robot_1_eight_ball' 'robot_2_four_ball' 'robot_3_stable_bot'
model_name = "resnet18" # "vgg""resnet18""resnet50"

data_root = f"./Image_Depth/{robot_name}"
os.makedirs(f"./checkpoint/{robot_name}", exist_ok=True)
txt_file = [f for f in os.listdir(data_root) if f.endswith(".txt")][0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# size = 16
g_s =(1,1)
piml = False #False # True


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
print(f"Robot: {robot_name}, Model: {model_name}, Grid Size: {g_s}, PIML: {piml}, focus feature dim {len(focus_functions)*g_s[0]*g_s[1]}")


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

# === 3. Feature Extraction ===
def extract_local_features(image, grid_size, focus_functions):
    segments = segment_image(image, grid_size)
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

        image_tensor = image.unsqueeze(0)  # add batch dim for focus measure
        focus_feat = process_tensor_image(image_tensor, grid_size=g_s)
        focus_feat_tensor = torch.tensor(focus_feat, dtype=torch.float32)

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

# Split data
train_df, temp_df = train_test_split(df, test_size=0.1, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_ds = DepthDataset(train_df)
val_ds = DepthDataset(val_df)
test_ds = DepthDataset(test_df)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=16, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=16, drop_last=True)

class PhysicsInformedResNet18(nn.Module):
    def __init__(self, physics_feat_dim=len(focus_functions)*g_s[0]*g_s[1], physics_hidden_dim=1):
        super(PhysicsInformedResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 输出 [B, 512, 1, 1]
        self.img_feat_dim = 512  
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_feat_dim, physics_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(physics_hidden_dim)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.img_feat_dim + physics_hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x_img, x_physics):
        x_img = self.backbone(x_img)           # [B, 512, 1, 1]
        x_img = torch.flatten(x_img, 1)        # [B, 512]
        x_physics = self.physics_encoder(x_physics)  # [B, physics_hidden_dim]
        x = torch.cat([x_img, x_physics], dim=1)     # [B, 512 + hidden]
        out = self.regressor(x)                      # [B, 1]
        return out.squeeze(1)                        # [B]

class PhysicsInformedResNet50(nn.Module):
    def __init__(self, physics_feat_dim=len(focus_functions)*g_s[0]*g_s[1], physics_hidden_dim=1): #len(focus_functions) * g_s[0]*g_s[1]
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

class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, x, physics_feat):
        x = self.conv(x)
        x = self.flat(x)
        return self.fc(x)

class VGG16Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1], nn.ReLU())
        self.regressor = nn.Linear(4096, 1)

    def forward(self, x, physics_feat):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.regressor(x)

class ResNet18Baseline(nn.Module):
    def __init__(self):
        super(ResNet18Baseline, self).__init__()
        resnet = models.resnet18(pretrained=True)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 1)
        self.model = resnet

    def forward(self, x, physics_feat):
        return self.model(x).squeeze(1)  # 输出为 [B]
        
class ResNet50Baseline(nn.Module):
    def __init__(self):
        super(ResNet50Baseline, self).__init__()
        resnet = models.resnet50(pretrained=True)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, 1)
        self.model = resnet

    def forward(self, x, physics_feat):
        return self.model(x).squeeze(1)  # [B]

def train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix, epochs=20):
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
            torch.save(model.state_dict(), f"{save_prefix}_best_grid{g_s[0]}.pth")
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
print(f"\n🚀 Training {model_name}...Grid Size {g_s[0],g_s[1]}")
if piml == True:
    # model = PhysicsInformedResNet18().to(device)
    model = PhysicsInformedResNet50().to(device)
elif model_name == "cnn":
    model = CNNBaseline().to(device)
elif model_name == "vgg":
    model = VGG16Baseline().to(device)
elif model_name == "resnet18":
    model = ResNet18Baseline().to(device)
elif model_name == "resnet50":
    model = ResNet50Baseline().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
save_prefix = f"checkpoint/{robot_name}/{robot_name}_{model_name}_grid{g_s[0]}"

train_model(model, train_loader, val_loader, optimizer, criterion, save_prefix)
model.load_state_dict(torch.load(f"{save_prefix}_best_grid{g_s[0]}.pth"))
mse, r2 = test_model(model, test_loader)

input_tensor = torch.randn(1, 3, 224, 224).to(device)
focus_tensor_test = torch.randn(1, len(focus_functions)*g_s[0]*g_s[1]).to(device)
macs, params = profile(model, inputs=(input_tensor,focus_tensor_test), verbose=False)

results.append({
    "MSE": mse,
    "R2": r2,
    "Params_M": round(params / 1e6, 2),
    "FLOPs_G": round(macs / 1e9, 2)
})
print(f"✅ MSE: {mse:.4f}, R^2: {r2:.4f}, Params_M: {round(params / 1e6, 2)}, FLOPs_G: {round(macs / 1e9, 2)}")

# Save results
output_csv = f"./checkpoint/{robot_name}/results_{robot_name}_{model_name}_grid{g_s[0]}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 Results saved to {output_csv}")