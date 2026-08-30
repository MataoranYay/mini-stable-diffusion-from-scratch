import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from pathlib import Path

class ImageTextDataset(Dataset):
    """
    Dataset for image-text pairs.

    Expected directory structure:
        data_dir/
            images/
                00001.jpg
                00002.jpg
                ...
            captions.json  # {"00001.jpg": "a photo of a cat", ...}

    Or with metadata.csv:
        file_name,text
        00001.jpg,a photo of a cat
        00002.jpg,a photo of a dog
    """
    def __init__(
        self,
        data_dir: str,
        image_size: int = 512,
        tokenizer=None,  # CLIP tokenizer
        max_length: int = 77,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        # Load captions
        self.samples = self._load_captions()
        print(f"📂 Loaded {len(self.samples)} image-text pairs from {data_dir}")

    def _load_captions(self):
        samples = []

        # Try JSON format
        json_path = self.data_dir / "captions.json"
        if json_path.exists():
            with open(json_path, "r") as f:
                captions = json.load(f)
            for img_name, caption in captions.items():
                img_path = self.data_dir / "images" / img_name
                if img_path.exists():
                    samples.append((str(img_path), caption))
            return samples

        # Try CSV format
        csv_path = self.data_dir / "metadata.csv"
        if csv_path.exists():
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_path = self.data_dir / "images" / row["file_name"]
                    if img_path.exists():
                        samples.append((str(img_path), row["text"]))
            return samples

        # Fallback: scan directory for image files and use filename as caption
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for img_path in (self.data_dir / "images").glob("*"):
            if img_path.suffix.lower() in image_extensions:
                caption = img_path.stem.replace("_", " ").replace("-", " ")
                samples.append((str(img_path), caption))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenize caption (if tokenizer provided, else return raw string)
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
            return {
                "image": image,
                "caption": caption,
            }


class DummyDataset(Dataset):
    """
    Dummy dataset for testing training code without real data.
    Generates random images and captions.
    """
    def __init__(self, num_samples=1000, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
        self.captions = [
            "a photo of a cat",
            "a beautiful sunset over the ocean",
            "a red car on the street",
            "an astronaut riding a horse",
            "a cyberpunk city at night",
            "a bowl of fresh fruit",
            "a mountain landscape with snow",
            "a portrait of a young woman",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image in [-1, 1]
        image = torch.randn(3, self.image_size, self.image_size)
        caption = self.captions[idx % len(self.captions)]

        return {
            "image": image,
            "caption": caption,
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: int = 512,
    tokenizer=None,
    use_dummy: bool = False,
):
    """Create a DataLoader for training."""
    if use_dummy:
        dataset = DummyDataset(num_samples=1000, image_size=image_size)
    else:
        dataset = ImageTextDataset(
            data_dir=data_dir,
            image_size=image_size,
            tokenizer=tokenizer,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
