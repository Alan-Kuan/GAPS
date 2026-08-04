

# GAPS: Comunicación Pub/Sub Consciente de GPU
Este repositorio contiene el marco de trabajo presentado en el artículo "Towards Low-Latency GPU-Aware Pub/Sub Communication for Real-Time Edge Computing", aceptado por RTCSA 2025.

## Implementaciones
Existen dos implementaciones de GAPS en diferentes ramas:

- `main`: GAPS-z, construido sobre Zenoh-cpp (con Zenoh-pico como backend)
- `iceoryx`: GAPS-i, construido sobre Iceoryx.

## Entorno
Se proporciona un entorno de Docker para desarrollo o pruebas.
Para configurarlo, asegúrese de cumplir con los siguientes requisitos:
- Docker está instalado
- Docker Compose está instalado
- NVIDIA Container Toolkit está instalado
    - Siga los pasos de instalación y configuración [aquí](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- El controlador NVIDIA de su equipo anfitrión es compatible con CUDA 12.6
- La GPU NVIDIA es compatible con las API del controlador CUDA como `cuMemCreate` y `cuMemExportToShareableHandle`

Existen dos entornos bajo `env/`:
- `x86`: para máquinas x86 con una GPU NVIDIA
- `jetson`: para sistemas embebidos NVIDIA Jetson como Jetson AGX Orin
    - :warning: Consulte [`env/jetson/README.md`](./env/jetson/README.md) para saber cómo construir su imagen base antes de continuar

Cambie al directorio correspondiente y luego ejecute los siguientes comandos:
```sh
docker compose up -d
ssh ubuntu@localhost -p 22222
```

Para destruir el entorno, ejecute el siguiente comando en el mismo directorio:
```sh
docker compose down
```

## Compilación
Para configurar las instrucciones de compilación con CMake y compilar el proyecto con Ninja,
ejecute los siguientes comandos en el directorio padre de `GAPS/`:
```sh
cmake GAPS -B build -G Ninja
ninja -C build
```

Los ejecutables compilados se generarán en `./build/src`.

**Opciones de compilación de CMake:**

- `PROFILING=[on|(off)]`: habilita o deshabilita el perfilado de la operación `put` del publicador y la devolución de llamada del suscriptor
- `BUILD_DEBUG=[on|(off)]`: habilita o deshabilita la compilación con código de depuración
- `BUILD_TORCH_SUPPORT=[on|(off)]`: habilita o deshabilita el soporte de PyTorch (es decir, compilar PyGAPS)
- `BUILD_EXAMPLES=[(on)|off]`: habilita o deshabilita la compilación de los códigos de ejemplo

> [!Note]
> El valor entre paréntesis es el predeterminado

## Pre-Commit
Pre-commit se utiliza para configurar ganchos pre-commit de clang-format.

1. Instale [pre-commit](https://pre-commit.com/) en su entorno virtual de Python.
2. Ejecute `pre-commit install` para instalar los ganchos.
3. Cada vez antes de realizar un commit, los archivos confirmados se formatearán con `clang-format`.

## Agradecimientos
Gracias a los siguientes trabajos por hacer posible este proyecto.

- Este proyecto depende de las siguientes bibliotecas de terceros:
    - [CUDA](https://developer.nvidia.com/cuda-toolkit)
    - [Zenoh-cpp](https://github.com/eclipse-zenoh/zenoh-cpp)
    - [Zenoh-pico](https://github.com/eclipse-zenoh/zenoh-pico)
    - [Iceoryx](https://github.com/eclipse-iceoryx/iceoryx)
    - [nanobind](https://github.com/wjakob/nanobind)
- Los códigos de ejemplo de este proyecto dependen de las siguientes bibliotecas de terceros:
    - [OpenCV](https://github.com/opencv/opencv)
    - [PyTorch](https://github.com/pytorch/pytorch)
    - [ultralytics](https://github.com/ultralytics/ultralytics)
- Gracias a [tlsf-bsd](https://github.com/jserv/tlsf-bsd) por mostrar cómo implementar el asignador TLSF.
- Gracias a [jetson-containers](https://github.com/dusty-nv/jetson-containers) por proporcionar contenedores de aprendizaje automático para sistemas embebidos NVIDIA Jetson.
