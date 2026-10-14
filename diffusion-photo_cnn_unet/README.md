# Diffusion Photo

启动 Streamlit 去噪演示：

```bash
python -m streamlit run app.py
```

在侧边栏选择 CNN 或 U-Net，然后上传 JPG/PNG 图片。页面会显示加噪图像和所选模型的重建结果。

使用 Docker 启动：

```bash
docker build -t diffusion-photo .
docker run --rm -p 8501:8501 diffusion-photo
```

然后访问 http://localhost:8501。镜像会包含 CNN 和 U-Net 的模型权重，并提供容器健康检查。

两种模型分别使用 `cnn_model.pt` 和 `model.pt` 权重。重新训练时，将训练图片命名为 `photo.jpg` 放在项目根目录，然后运行：

```bash
python train.py
```

默认依次训练 CNN 与 U-Net，并保存各自的权重文件。只训练 CNN 时可运行 `python train.py --model cnn`；只训练 U-Net 时运行 `python train.py --model unet`。