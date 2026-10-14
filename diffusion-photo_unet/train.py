import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import os

# 1. 定义处理彩色图像的 U-Net 模型
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


# 2. 扩散模型核心公式
def add_noise(x, alpha_t):
    """根据公式 z_t = sqrt(alpha_t)*x + sqrt(1-alpha_t)*noise 添加噪声"""
    noise = torch.randn_like(x)
    alpha_t_tensor = torch.as_tensor(alpha_t, dtype=x.dtype, device=x.device)
    z_t = torch.sqrt(alpha_t_tensor) * x + torch.sqrt(1 - alpha_t_tensor) * noise
    return z_t, noise

def reconstruct(z_t, predicted_noise, alpha_t):
    """根据预测的噪声逆向重建图像"""
    alpha_t_tensor = torch.as_tensor(alpha_t, dtype=z_t.dtype, device=z_t.device)
    x_recon = (z_t - torch.sqrt(1 - alpha_t_tensor) * predicted_noise) / torch.sqrt(alpha_t_tensor)
    return torch.clamp(x_recon, 0, 1) # 将像素值限制在合理范围内

# 3. 训练执行逻辑
if __name__ == "__main__":
    # 将图像统一缩放为 64x64 大小，平衡训练速度与特征保留
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    image_path = "photo.jpg" # 确保你的合照命名为 photo.jpg 且在同一目录下
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"未找到训练图片 {image_path}，请将彩色合照放在当前目录下。")
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0) # 增加批次维度 [1, 3, 64, 64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model = UNetDenoise().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 500
    print("开始训练模型...")
    for epoch in range(epochs):
        # 每次迭代使用随机的 alpha_t 增加模型的泛化能力
        alpha_t_rand = torch.rand(1).item() * 0.9 + 0.05
        z_t, true_noise = add_noise(x, alpha_t_rand)

        optimizer.zero_grad()
        predicted_noise = model(z_t)
        loss = criterion(predicted_noise, true_noise) # 计算预测噪声与真实噪声的误差
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, "model.pt")
    print("训练完成，模型已保存为 model.pt")