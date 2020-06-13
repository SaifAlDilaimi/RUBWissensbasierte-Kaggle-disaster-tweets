
# Docker
Build docker:

```bash
docker build -t wb_docker .
```

Run docker:

```bash
docker run --rm --gpus all --name wb_docker -it -v $(pwd):/tf -p 8888:8888 wb_docker
```