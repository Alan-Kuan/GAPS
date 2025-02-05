# Jetson Docker Environment

1. Install [jetson-containers](https://github.com/dusty-nv/jetson-containers) on Jetson AGX Orin.
    - See [this document](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) for more details.
2. Run the following command in the repo of jetson-containers to build the base image:
    ```sh
    PYTORCH_VERSION=2.5.0 jetson-containers build --skip-tests=intermediate torch torchvision
    ```
3. Then you can create the environment with Docker Compose like:
    ```sh
    docker compose up -d
    ```