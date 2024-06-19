FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    build-essential ninja-build \
    sudo curl wget git tar openssh-server

# cmake
WORKDIR /opt
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-`arch`.tar.gz"
RUN tar xvf cmake-3.29.6-linux-`arch`.tar.gz

# user
WORKDIR /
RUN useradd ubuntu -m -G sudo -s /usr/bin/bash
RUN passwd -d ubuntu
RUN echo 'export PATH="/opt/cmake-3.29.6-linux-`arch`/bin:/usr/local/cuda/bin:${PATH}"' >> /home/ubuntu/.bashrc

# rustup
WORKDIR /home/ubuntu
USER ubuntu
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.74.0 -y

# ssh server
WORKDIR /
USER root
RUN mkdir /var/run/sshd
RUN echo 'Port 22' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]