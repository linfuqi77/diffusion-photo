import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os

# 1. 定义用于处理彩色图像的 U-Net 模型
class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = _DoubleConv(3, 32)
        self.encoder2 = _DoubleConv(32, 64)
        self.encoder3 = _DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _DoubleConv(128, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = _DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = _DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = _DoubleConv(64, 32)
        self.output_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = self.upconv3(bottleneck)
        decoder3 = self.decoder3(torch.cat((decoder3, encoder3), dim=1))
        decoder2 = self.upconv2(decoder3)
        decoder2 = self.decoder2(torch.cat((decoder2, encoder2), dim=1))
        decoder1 = self.upconv1(decoder2)
        decoder1 = self.decoder1(torch.cat((decoder1, encoder1), dim=1))
        return self.output_conv(decoder1)


class CNNDenoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


# 2. 扩散模型核心公式
def add_noise(x, alpha_t):
    """根据 z_t = sqrt(alpha_t)*x + sqrt(1-alpha_t)*noise 的公式为图像添加噪声。"""
    noise = torch.randn_like(x)
    alpha_t_tensor = torch.as_tensor(alpha_t, dtype=x.dtype, device=x.device)
    z_t = torch.sqrt(alpha_t_tensor) * x + torch.sqrt(1 - alpha_t_tensor) * noise
    return z_t, noise

def reconstruct(z_t, predicted_noise, alpha_t):
    """根据预测噪声重建图像。"""
    alpha_t_tensor = torch.as_tensor(alpha_t, dtype=z_t.dtype, device=z_t.device)
    x_recon = (z_t - torch.sqrt(1 - alpha_t_tensor) * predicted_noise) / torch.sqrt(alpha_t_tensor)
    return torch.clamp(x_recon, 0, 1)  # 将像素值限制在有效范围内。

# 3. 训练执行逻辑
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN and/or U-Net denoising models.")
    parser.add_argument(
        "--model",
        choices=("cnn", "unet", "all"),
        default="all",
        help="Select which model to train (default: all).",
    )
    args = parser.parse_args()

    # 将图像统一缩放为 64x64，在训练速度和特征保留之间取得平衡。
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image_path = "photo.jpg"  # 将训练用彩色图片命名为 photo.jpg 并放在此目录下。
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Training image {image_path} not found. Place a color photo in the current directory.")
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0)  # 增加批次维度，形状为 [1, 3, 64, 64]。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    criterion = nn.MSELoss()
    epochs = 500

    model_options = {
        "cnn": (CNNDenoise, "cnn_model.pt", "CNN"),
        "unet": (UNetDenoise, "model.pt", "U-Net"),
    }
    selected_models = (
        model_options.values() if args.model == "all" else (model_options[args.model],)
    )
    for model_class, model_path, model_name in selected_models:
        model = model_class().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        print(f"Starting {model_name} training...")

        for epoch in range(epochs):
            alpha_t_rand = torch.rand(1).item() * 0.9 + 0.05
            z_t, true_noise = add_noise(x, alpha_t_rand)

            optimizer.zero_grad()
            predicted_noise = model(z_t)
            loss = criterion(predicted_noise, true_noise)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"{model_name} Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        torch.save({key: value.cpu() for key, value in model.state_dict().items()}, model_path)
        print(f"{model_name} training complete. Model saved as {model_path}")