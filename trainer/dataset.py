import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path

class ImageTextDataset(Dataset):
    """
    图像-文本对数据集。

    期望的数据集文件目录结构:
        data_dir/
            images/
                00001.jpg
                00002.jpg
                ...
            captions.json  # {"00001.jpg": "a photo of a cat", ...}
    """
    def __init__(
        self,
        data_dir: str,
        caption_name: str = "captions.json",
        image_size: int = 512,
        tokenizer=None,  # CLIP tokenizer
        max_length: int = 77,
    ):
        self.data_dir = Path(data_dir)
        self.caption_name = caption_name
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 图像裁剪+归一化
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        self.samples = self._load_captions()
        print(f"📂 Loaded {len(self.samples)} image-text pairs from {data_dir}")

    def _load_captions(self):
        samples = []

        # 读取 caption 文件，格式：
        # {
        #   "00001.jpg": "a photo of a cat",
        #   "00002.jpg": "a photo of a dog", 
        #   ...
        # }
        json_path = self.data_dir / self.caption_name
        if json_path.exists():
            with open(json_path, "r") as f:
                captions = json.load(f)
            for img_name, caption in captions.items():
                img_path = self.data_dir / "images" / img_name
                if img_path.exists():
                    samples.append((str(img_path), caption))
            return samples

        # 未找到 caption 文件，用文件名作为 caption
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for img_path in (self.data_dir / "images").glob("*"):
            if img_path.suffix.lower() in image_extensions:
                caption = img_path.stem.replace("_", " ").replace("-", " ")
                samples.append((str(img_path), caption))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取第 idx 个图像-文本对
        img_path, caption = self.samples[idx]

        # 加载并变换处理图像
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 编码提示词文本
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                [caption],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            return {
                "image": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "caption": caption,
            }
        else:
            # 如果没有提供分词器，则返回原始文本
            return {
                "image": image,
                "caption": caption,
            }