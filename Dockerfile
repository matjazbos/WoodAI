# ---- build stage ----
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# PyTorch from the CUDA 12.8 index
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Other Python packages from normal PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY CMakeLists.txt .
COPY draw_boxes.cpp .
COPY main.cpp .
COPY yolo_inference.py .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j

# ---- runtime stage ----
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-core-dev \
    libopencv-imgcodecs-dev \
    libopencv-imgproc-dev \
    python3 \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /work

COPY --from=builder /app/build/draw_boxes /usr/local/bin/draw_boxes
COPY --from=builder /app/build/wood_ai /usr/local/bin/wood_ai
COPY --from=builder /app/yolo_inference.py /usr/local/bin/yolo_inference.py

ENTRYPOINT ["wood_ai"]