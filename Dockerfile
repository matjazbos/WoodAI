# ---- build stage ----
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt .
COPY draw_boxes.cpp .
COPY main.cpp .

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j

# ---- runtime stage ----
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-core-dev \
    libopencv-imgcodecs-dev \
    libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY --from=builder /app/build/draw_boxes /usr/local/bin/draw_boxes
COPY --from=builder /app/build/wood_ai /usr/local/bin/wood_ai

ENTRYPOINT ["wood_ai"]