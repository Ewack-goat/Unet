import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import gradio as gr  # 导入 Gradio

from utils.data_loading import BasicDataset
from unet import UNet


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode="bilinear"
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    out = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)  # RGB 输出
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    color_map = {
        0: [0, 0, 0],  # 类别 0: 黑色
        1: [255, 0, 0],  # 类别 1: 红色
        2: [0, 255, 0],  # 类别 2: 绿色
        3: [0, 0, 255],  # 类别 3: 蓝色
        4: [255, 255, 0],  # 类别 4: 黄色
        5: [0, 255, 255],  # 类别 5: 青色
        6: [255, 0, 255],  # 类别 6: 品红色
    }

    for i in range(len(mask_values)):
        out[mask == i] = color_map.get(i, [255, 255, 255])  # 默认颜色为白色

    return Image.fromarray(out)


def create_combined_image(original, mask):
    original = original.convert("RGB")
    mask = mask.convert("L")
    combined = Image.new("RGB", (original.width + mask.width, original.height))
    combined.paste(original, (0, 0))
    combined.paste(mask.convert("RGB"), (original.width, 0))
    return combined


def segment_image(img):
    mask = predict_img(
        net=net,
        full_img=img,
        scale_factor=scale_factor,
        out_threshold=mask_threshold,
        device=device,
    )
    mask_image = mask_to_image(mask, mask_values)
    combined_image = create_combined_image(img, mask_image)
    return img, mask_image, combined_image


def main():
    global net, scale_factor, mask_threshold, mask_values, device

    model_path = r"C:\Users\22172\Downloads\checkpoint_epoch15.pth"
    scale_factor = 0.5
    mask_threshold = 0.5

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    net = UNet(n_channels=3, n_classes=7, bilinear=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading model {model_path}")
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    # 创建 Gradio 接口
    interface = gr.Interface(
        fn=segment_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=[
            gr.Image(type="pil", label="Original Image"),
            gr.Image(type="pil", label="Mask Image"),
            gr.Image(type="pil", label="Combined Image"),
        ],
        title="Image Segmentation",
        description="Upload an image to see the segmentation results.",
    )

    interface.launch()


if __name__ == "__main__":
    main()
