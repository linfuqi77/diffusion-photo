import torch
import pytest
import sys
import os

# 将根目录加入路径以便导入 train.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import DenoiseCNN, add_noise, reconstruct

@pytest.fixture
def sample_image_tensor():
    """生成一张模拟的 64x64 彩色图像张量用于测试"""
    return torch.rand(1, 3, 64, 64)

def test_add_noise_shape(sample_image_tensor):
    """测试：添加噪声后是否返回正确的 64x64 三通道图像"""
    alpha_t = 0.5
    z_t, noise = add_noise(sample_image_tensor, alpha_t)
    assert z_t.shape == (1, 3, 64, 64), "加噪后的图像尺寸不正确"
    assert noise.shape == (1, 3, 64, 64), "生成的噪声尺寸不正确"

def test_cnn_output_shape(sample_image_tensor):
    """测试：CNN 能否处理输入并产生预期形状的输出"""
    model = DenoiseCNN()
    output = model(sample_image_tensor)
    assert output.shape == (1, 3, 64, 64), "CNN 输出的特征图形状应与输入严格一致"

def test_reconstruct_valid_values(sample_image_tensor):
    """测试：重建函数是否返回有效的数值 (无 NaN，且在 0-1 之间)"""
    alpha_t = 0.5
    z_t, noise = add_noise(sample_image_tensor, alpha_t)
    
    model = DenoiseCNN()
    predicted_noise = model(z_t)
    x_recon = reconstruct(z_t, predicted_noise, alpha_t)
    
    assert not torch.isnan(x_recon).any(), "重建图像数值异常 (包含 NaN)"
    assert not torch.isinf(x_recon).any(), "重建图像数值异常 (包含 Inf)"
    assert torch.all(x_recon >= 0) and torch.all(x_recon <= 1), "重建图像像素值未被正确限制在 0-1 范围内"