import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from tqdm import tqdm
import cv2


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
    # 确定输出图像的尺寸
    out = np.zeros((mask.shape[-2], mask.shape[-1], 3), dtype=np.uint8)  # RGB 输出

    # 如果 mask 是多通道的，将其转换为单通道
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # 为每个类别指定颜色
    color_map = {
        0: [0, 0, 0],  # 类别 0: 黑色
        1: [255, 255, 255],  # 类别 1: 红色
        2: [255, 255, 255],  # 类别 2: 绿色
        3: [255, 255, 255],  # 类别 3: 蓝色
        4: [255, 255, 255],  # 类别 4: 黄色
        5: [255, 255, 255],  # 类别 5: 青色
        6: [255, 255, 255],  # 类别 6: 品红色
        # 添加其他类别的颜色
    }

    for i in range(len(mask_values)):
        out[mask == i] = color_map.get(i, [255, 255, 255])  # 默认颜色为白色


    return Image.fromarray(out)  # 返回图像


def create_combined_image(original, mask, alpha=0.5):
    # 将图像转换为RGB模式
    original = original.convert("RGBA")

    # 创建一个新的图像，用于叠加
    overlay = Image.new("RGBA", original.size)

    # 将 mask 转换为 RGB 模式
    mask = mask.convert("L")

    for x in range(original.width):
        for y in range(original.height):
            # 获取原图像素
            mask_value = mask.getpixel((x, y))
            if(mask_value == 0):
                overlay.putpixel((x, y), (0, 0, 0, 255))
            else:
                overlay.putpixel((x, y), original.getpixel((x, y)))


    # 将原图与 overlay 合成
    combined = Image.alpha_composite(original, overlay)

    return combined.convert("RGB")  # 转换为 RGB 模式


def process_video(
    input_video_path, output_video_path, net, scale_factor, mask_threshold, device, batch_size=4
):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    
    # 读取所有帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()  # 释放视频捕获

    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing Video"):
        batch_frames = frames[i:i + batch_size]

        # 将每帧转换为 PIL 图像
        img_batch = [Image.fromarray(frame) for frame in batch_frames]

        # 批量预测掩膜
        masks = [predict_img(net, img, device, scale_factor, mask_threshold) for img in img_batch]

        for j, mask in enumerate(masks):
            mask_image, most_frequent_class = mask_to_image(mask, [0, 1, 2, 3, 4, 5, 6])
            combined_image = create_combined_image(img_batch[j], mask_image)

            # 转换回 OpenCV 格式并写入视频
            combined_image_cv = cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGB2BGR)
            out.write(combined_image_cv)

    out.release()  # 释放视频写入对象


def main():
    model_path = "./checkpoints/best.pth"  # 指定模型路径
    input_dir = "./test"  # 输入文件夹
    output_dir = "./out"  # 输出文件夹
    scale_factor = 0.5
    mask_threshold = 0.5

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading model {model_path}")
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    mask_values = state_dict.pop("mask_values", [0, 1])
    net.load_state_dict(state_dict)

    logging.info("Model loaded!")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        
        if filename.endswith((".jpg", ".png")):  # 处理图像文件
            logging.info(f"Predicting image {filename} ...")

            img = Image.open(filepath)
            mask = predict_img(
                net=net,
                full_img=img,
                scale_factor=scale_factor,
                out_threshold=mask_threshold,
                device=device,
            )
            mask_image = mask_to_image(mask, mask_values)

            # 创建合并图像
            combined_image = create_combined_image(img, mask_image, alpha=0.5)

            # 保存输出文件
            out_filename = os.path.join(output_dir, f"{filename}.png")  # 使用PNG格式保存
            out_filename_mask = os.path.join(output_dir, f"{filename}_mask.png")
            mask_image.save(out_filename_mask)
            combined_image.save(out_filename)
            logging.info(f"Mask and overlay saved to {out_filename}")

        elif filename.endswith((".mp4", ".avi")):  # 处理视频文件
            logging.info(f"Processing video {filename} ...")
            input_video_path = os.path.join(input_dir, filename)
            output_video_path = os.path.join(output_dir, f"{filename}_output.avi")
            process_video(
                input_video_path,
                output_video_path,
                net,
                scale_factor,
                mask_threshold,
                device,
            )
            logging.info(f"Video processed and saved to {output_video_path}")

if __name__ == "__main__":
    main()