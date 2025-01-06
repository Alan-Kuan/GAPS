FROM nvcr.io/nvidia/pytorch:24.08-py3-igpu

ENV DEBIAN_FRONTEND=noninteractive

# dev dependencies
RUN apt update && apt install -y \
    build-essential ninja-build \
    sudo curl wget git tar tmux openssh-server

# optional dev dependencies
RUN apt update && apt install -y \
    libopencv-dev

# python packages
RUN python -m pip install --upgrade pip
RUN pip install nanobind==2.1.0 \
    opencv-python transformers

# zenoh
WORKDIR /root
RUN git clone https://github.com/eclipse-zenoh/zenoh-pico.git --branch 1.1.0 --depth 1
WORKDIR /root/zenoh-pico
RUN cmake -B build -G Ninja -DFRAG_MAX_SIZE=17000000
RUN ninja -C build && ninja -C build install

WORKDIR /root
RUN git clone https://github.com/eclipse-zenoh/zenoh-cpp.git --branch 1.1.0 --depth 1
WORKDIR /root/zenoh-cpp
RUN cmake -B build -DZENOHCXX_ZENOHC=off -DZENOHCXX_ZENOHPICO=on
RUN make -C build install

# user
WORKDIR /
RUN useradd ubuntu -m -G sudo -s /usr/bin/bash
RUN passwd -d ubuntu
RUN echo 'export PATH="/usr/local/cuda/bin:${PATH}"' >> /home/ubuntu/.bashrc
RUN echo 'export PYTHONPATH=./build/src' >> /home/ubuntu/.bashrc

# ssh server
WORKDIR /
USER root
RUN mkdir /var/run/sshd
RUN echo 'Port 22' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
