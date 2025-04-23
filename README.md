# GAPS: GPU-Aware Pub/Sub Communication

## Environment
A Docker development environment is priovded in `env/`.
To setup the environment, make sure the following requirements are met:
- Docker is installed
- Docker Compose is installed
- NVIDIA Container Toolkit is installed
    - Follow the installation and configuration steps [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- CUDA 12.6 is supported by your host's NVIDIA driver
- Some CUDA Driver APIs like `cuMemCreate` and `cuMemExportToShareableHandle` are supported by your NVIDIA GPU

There is an environment for development on x86 machine in `env/dev` and an environment for testing on Jetson Dev Board in `env/jetson`.

Choose one environment, and then run the following commands:
```sh
docker compose up -d
ssh ubuntu@localhost -p 22222
```

To destroy the environment, run the following command:
```sh
docker compose down
```

## Build
To configure build instructions with CMake and build the project with Ninja,
run the following commands in the parent directory of `GAPS/`:
```sh
cmake GAPS -B build -G Ninja
ninja -C build
```

After building the project, the compiled executables lie in `build/src`.

### CMake Build Options
- `PROFILING=[on|(off)]`: whether to profile publisher's put and subscriber's callback
    > [!WARN]
    > When profiling the callback,
    > make sure the publishing interval is high enough,
    > or the outputing lines will affect the time measurement significantly
- `BUILD_DEBUG=[on|(off)]`: whether to build with debugging codes
- `BUILD_TORCH_SUPPORT=[on|(off)]`: whether to build PyTorch support
- `BUILD_EXAMPLES=[(on)|off]`: whether to build example codes

> [!Note]
> The one wrapped with parentheses is the default value

## Pre-Commit
Pre-commit is used to setup clang-format pre-commit hooks.

1. Install [pre-commit](https://pre-commit.com/) in your Python virtual environment.
2. Run `pre-commit install` to install the hooks.
3. Each time before committing, committed files will be formatted with `clang-format`.

## Acknowledgements
Thanks for the following works to make this project possible.

- This project depends on the following third-party libraries:
    - [CUDA](https://developer.nvidia.com/cuda-toolkit)
    - [Zenoh-cpp](https://github.com/eclipse-zenoh/zenoh-cpp)
    - [Zenoh-pico](https://github.com/eclipse-zenoh/zenoh-pico)
    - [Iceoryx](https://github.com/eclipse-iceoryx/iceoryx)
    - [nanobind](https://github.com/wjakob/nanobind)
- Example codes in this projects depend on the following third-party libraries:
    - [OpenCV](https://github.com/opencv/opencv)
    - [PyTorch](https://github.com/pytorch/pytorch)
    - [ultralytics](https://github.com/ultralytics/ultralytics)
- Thanks [tlsf-bsd](https://github.com/jserv/tlsf-bsd) for showing how to implement the TLSF allocator.
- Thanks [jetson-containers](https://github.com/dusty-nv/jetson-containers) for providing machine learning containers on NVIDIA Jetson embedded systems.