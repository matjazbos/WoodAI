# Wood AI
This project trains a YOLO-based object detector to identify wood knots in individual board-frame images using YOLO-format bounding box annotations. The training pipeline is implemented in Python with PyTorch/Ultralytics and runs in Docker with CUDA support, while the final deployment target is a C++ application built with CMake. The model is exported to ONNX for later inference in C++, where board-level processing and stitching logic are handled separately from training.

## Build
    docker build -t wood-ai .

## Run inference
    docker run --rm --gpus all --shm-size=2g --entrypoint python --mount type=bind,src="${PWD}",target=/workdir wood-ai -u /workdir/yolo_inference.py --data-root /workdir/WoodDataset --work-dir /tmp/yolo_dataset --output /workdir/output/runs --batch-size 40 --epochs 30 --workers 4


## Run classifier
### For all boards:
    docker run --rm --mount type=bind,src="${PWD}",target=/workdir wood-ai 

### For one board:
    docker run --rm --mount type=bind,src="${PWD}",target=/workdir wood-ai <board id>


## Results
I ran the inference with the following parameters:
- Base model: **yolo11n.pt**
- Batch size: **40**
- Number of epochs: **30**
- Number of workers: **4**
- Input image size: **640**
- Validation split ratio: **0.1**

### Some metrics and interpretations
- **Precision: 0.886:** means that about 88.6% of the predicted knot boxes are correct, so the model does not produce many false positives
- **Recall: 0.877:** means it finds about 87.7% of the real knots, so it misses relatively few
- **mAP50: 0.926** is very good and shows the model is reliably detecting knots
- **mAP50-95: 0.647** is the stricter metric, so this lower value is normal. It suggests detection is strong, but box placement could still be improved somewhat

Overall, the model seems reliable and well-trained. It is good at both finding knots and avoiding incorrect detections. The main room for improvement is making the predicted rectangles tighter and more precise.
