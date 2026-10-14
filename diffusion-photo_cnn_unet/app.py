import streamlit as st
import torch
from io import BytesIO
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from train import CNNDenoise, UNetDenoise, add_noise, reconstruct

st.set_page_config(page_title="Diffusion Photo PoC", layout="wide")
st.title("Diffusion Denoising Demo")

# 交互参数设置
alpha_t = st.sidebar.slider("Original Signal Retention (alpha_t)", min_value=0.1, max_value=0.99, value=0.5)
model_name = st.sidebar.radio("Denoising model", ("CNN", "U-Net"))
model_class, model_filename = {
    "CNN": (CNNDenoise, "cnn_model.pt"),
    "U-Net": (UNetDenoise, "model.pt"),
}[model_name]

uploaded_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns(3)
    
    # 1. 预处理原始图像
    original_image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    x = transform(original_image).unsqueeze(0)
    
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    # 2. 生成加噪图像
    z_t, true_noise = add_noise(x, alpha_t)
    z_t_display = transforms.ToPILImage()(z_t.squeeze(0).clamp(0, 1))
    
    with col2:
        st.image(z_t_display, caption=f"Noisy Image (alpha={alpha_t})", use_container_width=True)
    
    # 3. 加载模型并重建图像
    model_path = Path(__file__).resolve().with_name(model_filename)
    with col3:
        if not model_path.exists() or model_path.stat().st_size == 0:
            st.error(f"No valid weights found at `{model_filename}`. Add `photo.jpg` and run `python train.py` to train both models.")
        else:
            try:
                model = model_class()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()

                with torch.no_grad():
                    predicted_noise = model(z_t)
                    x_recon = reconstruct(z_t, predicted_noise, alpha_t)

                mse = torch.nn.functional.mse_loss(x_recon, x).item()
                recon_display = transforms.ToPILImage()(x_recon.squeeze(0))
                st.image(recon_display, caption=f"{model_name} Reconstruction", use_container_width=True)
                image_buffer = BytesIO()
                recon_display.save(image_buffer, format="PNG")
                st.download_button(
                    f"Download {model_name} Reconstruction",
                    data=image_buffer.getvalue(),
                    file_name=f"{model_name.lower()}_reconstruction.png",
                    mime="image/png",
                )
                st.metric("Reconstruction MSE", f"{mse:.6f}")
            except Exception as error:
                st.error(f"Could not load or run `model.pt`: {error}")