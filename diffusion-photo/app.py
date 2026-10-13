import streamlit as st
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from train import DenoiseCNN, add_noise, reconstruct

st.set_page_config(page_title="Diffusion Photo PoC", layout="wide")
st.title("扩散去噪模型演示")

# 侧边栏交互参数
alpha_t = st.sidebar.slider("保留原始信号的比例 (Alpha_t)", min_value=0.1, max_value=0.99, value=0.5)

uploaded_file = st.file_uploader("请上传你们的合照 (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    
    # 1. 预处理原图
    original_image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    x = transform(original_image).unsqueeze(0)
    
    with col1:
        st.image(original_image, caption="原始上传合照", use_container_width=True)
    
    # 2. 生成加噪图像
    z_t, true_noise = add_noise(x, alpha_t)
    z_t_display = transforms.ToPILImage()(z_t.squeeze(0).clamp(0, 1))
    
    with col2:
        st.image(z_t_display, caption=f"加噪图像 (Alpha={alpha_t})", use_container_width=True)
    
    # 3. 加载模型进行重建
    model_path = Path(__file__).resolve().with_name("model.pt")
    with col3:
        if not model_path.exists() or model_path.stat().st_size == 0:
            st.error("未找到有效的模型参数 `model.pt`。请先准备训练图片并运行 `python train.py`。")
        else:
            try:
                model = DenoiseCNN()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()

                with torch.no_grad():
                    predicted_noise = model(z_t)
                    x_recon = reconstruct(z_t, predicted_noise, alpha_t)

                mse = torch.nn.functional.mse_loss(x_recon, x).item()
                recon_display = transforms.ToPILImage()(x_recon.squeeze(0))
                st.image(recon_display, caption="CNN 重建结果", use_container_width=True)
                st.metric("重建 MSE", f"{mse:.6f}")
            except Exception as error:
                st.error(f"无法加载或运行 `model.pt`：{error}")