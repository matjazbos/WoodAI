# Wood AI
This project trains a YOLO-based object detector to identify wood knots in individual board-frame images using YOLO-format bounding box annotations. The training pipeline is implemented in Python with PyTorch/Ultralytics and runs in Docker with CUDA support, while the final deployment target is a C++ application built with CMake. The model is exported to ONNX for later inference in C++, where board-level processing and stitching logic can be handled separately from training.

## Build
    docker build -t wood-ai .

## Run inference

### Testing
    docker run --rm --gpus all --shm-size=2g --entrypoint python --mount type=bind,src="${PWD}",target=/workdir wood-ai -u /workdir/yolo_inference.py --data-root /workdir/WoodDataset --work-dir /tmp/yolo_dataset --output /workdir/output/runs --batch-size 40 --epochs 30 --workers 4

### Final image
    docker run --rm --gpus all --shm-size=2g --entrypoint python --mount type=bind,src="${PWD}",target=/workdir wood-ai -u /usr/local/bin/yolo_inference.py --data-root /workdir/WoodDataset --work-dir /tmp/yolo_dataset --output /workdir/output/runs --batch-size 40 --epochs 30 --workers 4