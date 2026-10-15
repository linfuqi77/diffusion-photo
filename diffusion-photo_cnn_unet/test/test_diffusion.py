import torch
import pytest
import sys
import os

# 将项目根目录加入路径，以便导入 train.py。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import (
    CNNDenoise,
    UNetDenoise,
    add_noise,
    fit_linear_noise_regression,
    predict_linear_noise,
    reconstruct,
)

@pytest.fixture
def sample_image_tensor():
    """生成一个用于测试的 64x64 彩色图像张量。"""
    return torch.rand(1, 3, 64, 64)

def test_add_noise_shape(sample_image_tensor):
    """测试添加噪声后是否保留 64x64 RGB 图像的形状。"""
    alpha_t = 0.5
    z_t, noise = add_noise(sample_image_tensor, alpha_t)
    assert z_t.shape == (1, 3, 64, 64), "Noisy image has an unexpected shape."
    assert noise.shape == (1, 3, 64, 64), "Generated noise has an unexpected shape."

def test_unet_output_shape(sample_image_tensor):
    """测试 U-Net 是否返回预期形状的输出。"""
    model = UNetDenoise()
    output = model(sample_image_tensor)
    assert output.shape == (1, 3, 64, 64), "U-Net output shape should match the input shape."

def test_cnn_output_shape(sample_image_tensor):
    """测试 CNN 是否返回与输入相同形状的噪声预测。"""
    model = CNNDenoise()
    output = model(sample_image_tensor)
    assert output.shape == (1, 3, 64, 64), "CNN output shape should match the input shape."

def test_linear_regression_output_shape(sample_image_tensor):
    """测试线性回归基线可以拟合并预测与输入相同形状的噪声。"""
    coefficients = fit_linear_noise_regression(sample_image_tensor, num_samples=2)
    assert coefficients.shape == (5, 3)
    noisy_image, _ = add_noise(sample_image_tensor, 0.5)
    predicted_noise = predict_linear_noise(noisy_image, 0.5, coefficients)
    assert predicted_noise.shape == sample_image_tensor.shape
    assert torch.isfinite(predicted_noise).all()

def test_reconstruct_valid_values(sample_image_tensor):
    """测试重建结果是否为有限值，且数值范围在 [0, 1] 之间。"""
    alpha_t = 0.5
    z_t, noise = add_noise(sample_image_tensor, alpha_t)
    
    model = UNetDenoise()
    predicted_noise = model(z_t)
    x_recon = reconstruct(z_t, predicted_noise, alpha_t)
    
    assert not torch.isnan(x_recon).any(), "Reconstructed image contains NaN values."
    assert not torch.isinf(x_recon).any(), "Reconstructed image contains infinite values."
    assert torch.all(x_recon >= 0) and torch.all(x_recon <= 1), "Reconstructed pixels should be in the range [0, 1]."