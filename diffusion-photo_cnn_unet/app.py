import streamlit as st
import torch
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from train import (
    CNNDenoise,
    UNetDenoise,
    add_noise,
    predict_linear_noise,
    reconstruct,
)

st.set_page_config(page_title="Diffusion Photo PoC", layout="wide")
st.title("Diffusion Denoising Demo")

# 交互参数设置
alpha_t = st.sidebar.slider("Original Signal Retention (alpha_t)", min_value=0.1, max_value=0.95, value=0.5)

uploaded_file = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. 预处理原始图像
    original_image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    x = transform(original_image).unsqueeze(0)

    input_col, noisy_col = st.columns(2)
    with input_col:
        st.image(original_image, caption="Original Image", use_container_width=True)

    # 2. 生成加噪图像
    z_t, _ = add_noise(x, alpha_t)
    z_t_display = transforms.ToPILImage()(z_t.squeeze(0).clamp(0, 1))

    with noisy_col:
        st.image(z_t_display, caption=f"Noisy Image (alpha={alpha_t})", use_container_width=True)

    st.subheader("Linear baseline vs CNN vs U-Net")
    st.caption(
        "All three methods process the same noisy image at the same alpha_t. "
        "Lower MSE means a closer reconstruction for this sample."
    )
    comparison_columns = st.columns(3)
    scores = {}

    with comparison_columns[0]:
        st.markdown("### Linear Regression")
        baseline_path = Path(__file__).resolve().with_name("linear_baseline.npz")
        if not baseline_path.exists() or baseline_path.stat().st_size == 0:
            st.error(
                "No trained linear baseline found. Run `python train.py --model baseline` "
                "with `photo.jpg` in the project directory."
            )
        else:
            try:
                with np.load(baseline_path) as baseline_data:
                    coefficients = baseline_data["coefficients"].copy()
                with torch.no_grad():
                    predicted_noise = predict_linear_noise(z_t, alpha_t, coefficients)
                    x_recon = reconstruct(z_t, predicted_noise, alpha_t)

                mse = torch.nn.functional.mse_loss(x_recon, x).item()
                scores["Linear Regression"] = mse
                recon_display = transforms.ToPILImage()(x_recon.squeeze(0))
                st.image(recon_display, caption="Linear Regression Reconstruction", use_container_width=True)
                image_buffer = BytesIO()
                recon_display.save(image_buffer, format="PNG")
                st.download_button(
                    "Download Linear Regression Reconstruction",
                    data=image_buffer.getvalue(),
                    file_name="linear_regression_reconstruction.png",
                    mime="image/png",
                )
                st.metric("Reconstruction MSE (lower is better)", f"{mse:.6f}")
            except Exception as error:
                st.error(f"Could not load or run `linear_baseline.npz`: {error}")

    model_options = (
        ("CNN", CNNDenoise, "cnn_model.pt"),
        ("U-Net", UNetDenoise, "model.pt"),
    )

    for (model_name, model_class, model_filename), result_col in zip(
        model_options, comparison_columns[1:]
    ):
        with result_col:
            st.markdown(f"### {model_name}")
            model_path = Path(__file__).resolve().with_name(model_filename)
            if not model_path.exists() or model_path.stat().st_size == 0:
                st.error(
                    f"No valid weights found at `{model_filename}`. "
                    "Add `photo.jpg` and run `python train.py` to train both models."
                )
                continue

            try:
                model = model_class()
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()

                with torch.no_grad():
                    predicted_noise = model(z_t)
                    x_recon = reconstruct(z_t, predicted_noise, alpha_t)

                mse = torch.nn.functional.mse_loss(x_recon, x).item()
                scores[model_name] = mse
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
                st.metric("Reconstruction MSE (lower is better)", f"{mse:.6f}")
            except Exception as error:
                st.error(f"Could not load or run `{model_filename}`: {error}")

    st.subheader("Result analysis")
    if len(scores) == 3:
        baseline_mse = scores["Linear Regression"]
        best_model = min(("CNN", "U-Net"), key=scores.__getitem__)
        best_mse = scores[best_model]
        if baseline_mse > 0:
            improvement = (baseline_mse - best_mse) / baseline_mse * 100
            if improvement > 0:
                st.success(
                    f"{best_model} has the lowest MSE for this image/noise sample "
                    f"({best_mse:.6f}), {improvement:.2f}% lower than the linear "
                    f"regression baseline ({baseline_mse:.6f})."
                )
            else:
                st.warning(
                    f"Neither CNN nor U-Net beats the linear regression baseline on this "
                    f"sample. The best neural model is {best_model} "
                    f"(MSE {best_mse:.6f}); baseline MSE is {baseline_mse:.6f}."
                )
        else:
            st.info(f"{best_model} has the lowest MSE for this sample ({best_mse:.6f}).")

        st.caption(
            "This conclusion applies only to the current image and noise draw. "
            "The models and linear baseline are trained from one photo.jpg, so this "
            "single-sample result does not establish general performance."
        )
    else:
        st.info("Analysis will appear after valid weights for all three methods are available.")