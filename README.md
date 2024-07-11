# shoz

## Environment
A Docker development environment is priovded in `env/`.
To setup the environment, make sure the following requirements are met:
- Docker is installed
- Docker Compose is installed
- NVIDIA Container Toolkit is installed
    - Follow the installation and configuration steps [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- CUDA 12.3 is supported by your host's NVIDIA driver
- Some CUDA Driver APIs like `cuMemCreate` and `cuMemExportToShareableHandle` are supported by your NVIDIA GPU

```sh
cd env
docker compose up -d
ssh ubuntu@localhost -p 22222
```

To destroy the environment, run the following command:
```sh
docker compose down
```

## Build
Configure build instructions with CMake and build the project with Ninja.
Run the following commands in the home directory:
```sh
cmake shoz -B build -G Ninja -DZENOHC_CARGO_CHANNEL=1.74.0
ninja -C build
```

After building the project, the compiled executables lie in `build/src`.

## Pre-Commit
Pre-commit is used to setup clang-format pre-commit hooks.

1. Install [pre-commit](https://pre-commit.com/) in your Python virtual environment.
2. Run `pre-commit install` to install the hooks.
3. Each time before committing, committed files will be formatted with `clang-format`.