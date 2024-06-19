# shoz

## Environment
A Docker development environment is priovded in `env/`.
To setup the environment, make sure the following requirements are met:
- Docker is installed
- Docker Compose is installed
- NVIDIA Container Toolkit is installed
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
Clone the repo in the development environment.
Then, configure build instructions with CMake and build the project with Ninja.
```sh
git clone https://github.com/Alan-Kuan/shoz
cmake -B build -G Ninja -DZENOHC_CARGO_CHANNEL=1.74.0
ninja -C build
```

After building the project, the compiled executable lies in `build/src`.