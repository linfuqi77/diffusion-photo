# Diffusion Photo

启动 Streamlit 去噪演示：

```bash
python -m streamlit run app.py
```

上传 JPG/PNG 图片后，页面会使用相同的输入图、加噪样本和 `alpha_t` 并排比较线性回归基线、CNN 与 U-Net 的重建图像和 MSE。页面也会说明本次误差最低的方法及其相对线性回归基线的变化。

使用 Docker 启动：

```bash
docker build -t diffusion-photo .
docker run --rm -p 8501:8501 diffusion-photo
```

然后访问 http://localhost:8501。镜像会包含 CNN 和 U-Net 的模型权重，并提供容器健康检查。

模型权重和线性回归参数分别保存在 `cnn_model.pt`、`model.pt` 和 `linear_baseline.npz`。重新训练时，将训练图片命名为 `photo.jpg` 放在项目根目录，然后运行：

```bash
python train.py
```

默认拟合线性回归基线，然后依次训练 CNN 与 U-Net。仅生成/更新线性回归基线时可运行 `python train.py --model baseline`；训练 CNN 或 U-Net 时可分别运行 `python train.py --model cnn` 或 `python train.py --model unet`，这些命令也会更新线性回归基线。

线性回归以加噪 RGB 像素和 `alpha_t` 为特征预测噪声。应用中三种方法使用相同的加噪张量，并以原始上传图像计算重建 MSE。报告只解释当前图片和当前随机噪声的一次结果；由于此示例训练使用单张 `photo.jpg`，不能将单次对比视为模型在一般照片上的性能结论。