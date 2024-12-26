# shoz

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
run the following commands in the parent directory of `shoz/`:
```sh
cmake shoz -B build -G Ninja
ninja -C build
```

After building the project, the compiled executables lie in `build/src`.

### CMake Build Options
- `BUILD_TORCH_SUPPORT=[on|(off)]`: whether to build PyTorch support
- `BUILD_EXAMPLES=[(on)|off]`: whether to build example codes
- `CMAKE_BUILD_TYPE=[(Release)|Debug]`: whether to build with debug flags (in our project, we also add some checking codes if build type is not "Release")

> [!Note]
> The one wrapped with parentheses is the default value

## Pre-Commit
Pre-commit is used to setup clang-format pre-commit hooks.

1. Install [pre-commit](https://pre-commit.com/) in your Python virtual environment.
2. Run `pre-commit install` to install the hooks.
3. Each time before committing, committed files will be formatted with `clang-format`.