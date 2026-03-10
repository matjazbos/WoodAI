# Wood AI


## Build
docker build -t wood-ai .

## Run inference
docker run --rm --gpus all --entrypoint python --mount type=bind,src="${PWD}",target=/workdir wood-ai /usr/local/bin/inference.py --data-root /workdir/WoodDataset
docker run --rm --gpus all --entrypoint python --mount type=bind,src="${PWD}",target=/workdir wood-ai /workdir/inference.py --data-root /workdir/WoodDataset