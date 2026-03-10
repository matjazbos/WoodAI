import argparse
import random
from pathlib import Path

import torch
from ultralytics import YOLO


def build_yolo_dataset(data_root: Path, work_dir: Path, val_ratio: float, seed: int):
    """
    Create a YOLO dataset config without copying images/labels.

    It writes:
      work_dir/
        train.txt
        val.txt
        dataset.yaml

    Each txt file contains absolute image paths.
    Labels stay in the original labels folder with matching names.
    """

    images_dir = data_root / "images"
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {images_dir}")

    image_paths = list(image_paths)
    random.Random(seed).shuffle(image_paths)

    val_count = max(1, int(len(image_paths) * val_ratio))
    val_images = image_paths[:val_count]
    train_images = image_paths[val_count:]

    if not train_images:
        raise RuntimeError(
            "Validation split is too large; no training images remain.")

    work_dir.mkdir(parents=True, exist_ok=True)

    train_txt = work_dir / "train.txt"
    val_txt = work_dir / "val.txt"

    train_txt.write_text(
        "\n".join(str(p.as_posix()) for p in train_images) + "\n",
        encoding="utf-8",
    )
    val_txt.write_text(
        "\n".join(str(p.as_posix()) for p in val_images) + "\n",
        encoding="utf-8",
    )

    yaml_path = work_dir / "dataset.yaml"
    yaml_text = f"""train: {train_txt.as_posix()}
val: {val_txt.as_posix()}

names:
  0: knot
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()

    # Root folder with:
    #   WoodDataset/images
    #   WoodDataset/labels
    parser.add_argument("--data-root", type=str, default="WoodDataset")

    # Where to create the YOLO-formatted train/val dataset.
    parser.add_argument("--work-dir", type=str, default="yolo_dataset")

    # Where Ultralytics should write training runs and checkpoints.
    parser.add_argument("--output", type=str, default="runs")

    # Base YOLO model to start from.
    # Example small models:
    #   yolo11n.pt
    #   yolo11s.pt
    #
    # Smaller model = faster training/inference, lower capacity.
    parser.add_argument("--model", type=str, default="yolo11n.pt")

    # Number of full passes over the training set.
    parser.add_argument("--epochs", type=int, default=20)

    # Input image size used by YOLO during training.
    # Smaller size = faster, but may reduce accuracy.
    parser.add_argument("--imgsz", type=int, default=640)

    # Batch size used during training.
    # Increase if you have enough GPU memory.
    parser.add_argument("--batch-size", type=int, default=8)

    # Fraction of images reserved for validation.
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Random seed for reproducibility.
    parser.add_argument("--seed", type=int, default=42)

    # Device selection for Ultralytics.
    # Common values:
    #   "0"   -> first GPU
    #   "cpu" -> CPU only
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output)

    # Set random seeds for reproducible splitting and behavior.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Print CUDA diagnostics so it is obvious whether GPU is available.
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_version={torch.version.cuda}", flush=True)
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    # Build the temporary YOLO dataset structure and generate dataset.yaml.
    yaml_path = build_yolo_dataset(
        data_root=data_root,
        work_dir=work_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"dataset_yaml={yaml_path}", flush=True)

    # Load a pretrained YOLO model.
    #
    # This does not yet train the model. It just creates the model object
    # initialized from the checkpoint given by --model.
    model = YOLO(args.model)

    # Train the detector.
    #
    # Important:
    # - data: dataset.yaml path
    # - project: base output directory
    # - name: subfolder name inside project
    #
    # Final training outputs will usually end up in:
    #   runs/wood_knots/
    #
    # with weights in:
    #   runs/wood_knots/weights/
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(output_dir),
        name="wood_knots",
        exist_ok=True,
    )

    # After training, Ultralytics stores the best checkpoint here by default.
    best_pt = output_dir / "wood_knots" / "weights" / "best.pt"
    print(f"best_pt={best_pt}", flush=True)

    # Export the trained checkpoint to ONNX for later use in C++.
    #
    # This creates an ONNX model next to the trained weights, typically:
    #   best.onnx
    #
    # That ONNX file is the artifact you will likely want for C++ inference.
    exported = YOLO(str(best_pt)).export(format="onnx", opset=20)
    print(f"exported={exported}", flush=True)


if __name__ == "__main__":
    main()
