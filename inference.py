import argparse
import random
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    Convert a YOLO-format box:
        (x_center, y_center, width, height) in normalized coordinates
    into pixel coordinates:
        (x1, y1, x2, y2)

    The returned coordinates are clamped so they stay inside the image.
    """
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h

    # Clamp coordinates to image boundaries
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(0.0, min(x2, img_w - 1))
    y2 = max(0.0, min(y2, img_h - 1))
    return [x1, y1, x2, y2]


class WoodDataset(Dataset):
    """
    Custom dataset for images in:
        WoodDataset/images/*.png
    and matching labels in:
        WoodDataset/labels/*.txt

    Each label file contains YOLO-format boxes:
        <class_id> <x_center> <y_center> <width> <height>
    """

    def __init__(self, root):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        # Collect all PNG image files
        self.image_paths = sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise RuntimeError(f"No PNG images found in {self.images_dir}")

    def __len__(self):
        # Number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Resolve image path and matching label path
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        boxes = []
        labels = []

        # Parse YOLO label file if it exists
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        continue

                    class_id = int(float(parts[0]))
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Convert normalized YOLO box into pixel corner coordinates
                    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)

                    # Only keep valid boxes with non-zero area
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])

                        # Faster R-CNN uses class 0 as background,
                        # so foreground classes must start at 1
                        labels.append(class_id + 1)

        # Convert annotations into tensors expected by torchvision detection models
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Empty targets for images without any objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Convert PIL image into PyTorch tensor in [0, 1]
        image_tensor = F.to_tensor(image)

        # Detection models expect a target dictionary with these fields
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "path": str(img_path),  # not used by model, but useful for debugging
        }

        return image_tensor, target


def collate_fn(batch):
    """
    Custom collate function needed for torchvision detection models.
    They expect a list of images and a list of targets, not stacked tensors.
    """
    return tuple(zip(*batch))


def build_model(num_classes):
    """
    Create a Faster R-CNN model with a ResNet50-FPN backbone.

    num_classes includes background, so:
        1 foreground class -> num_classes = 2
    """
    # weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)

    # Replace the classification head to match the number of classes in this task
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Run one training epoch and return average training loss.
    """
    model.train()
    running_loss = 0.0

    for step, (images, targets) in enumerate(loader, start=1):
        # Move data to GPU/CPU
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"].to(device),
                "area": t["area"].to(device),
                "iscrowd": t["iscrowd"].to(device),
            }
            for t in targets
        ]

        # Detection models return a dictionary of losses during training
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print occasional progress
        if step % 10 == 0 or step == len(loader):
            print(f"epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    """
    Run one validation epoch and return average validation loss.

    For torchvision detection models, losses are produced in train mode
    when targets are provided, so validation is done with model.train()
    inside torch.no_grad().
    """
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        # Move data to GPU/CPU
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "image_id": t["image_id"].to(device),
                "area": t["area"].to(device),
                "iscrowd": t["iscrowd"].to(device),
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="WoodDataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=1, help="foreground classes, excluding background")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Pick GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_version={torch.version.cuda}")
        print(f"gpu={torch.cuda.get_device_name(0)}")

    # return

    # Build dataset from folder structure
    dataset = WoodDataset(args.data_root)

    # Split into train/validation subsets
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset too small for the requested validation split.")

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Build detection model
    # +1 because torchvision includes background as class 0
    model = build_model(num_classes=args.num_classes + 1)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Checkpoint output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    # Main training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate_one_epoch(model, val_loader, device)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Save last checkpoint every epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": args.num_classes,
        }

        torch.save(checkpoint, output_dir / "last.pt")

        # Save best checkpoint based on validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")
            print(f"saved best model to {output_dir / 'best.pt'}")

    return
    # Export the best model to ONNX

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ONNX export works best on CPU
    model.cpu()

    # Dummy input with the same shape as your training images
    sample_image, _ = dataset[0]
    dummy_input = sample_image.unsqueeze(0).cpu()

    torch.onnx.export(
        model,
        dummy_input,
        output_dir / "best.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )

    print(f"saved ONNX model to {output_dir / 'best.onnx'}")


if __name__ == "__main__":
    main()