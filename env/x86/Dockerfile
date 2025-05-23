FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

#
#  Tools
#
RUN apt update && apt install -y \
    build-essential ninja-build \
    sudo curl wget git tar tmux vim openssh-server \
    python3-pip

# install CMake
WORKDIR /opt
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-`arch`.tar.gz"
RUN tar xvf cmake-3.29.6-linux-`arch`.tar.gz

#
#  Dependencies
#

# install Zenoh-pico
WORKDIR /root
RUN git clone https://github.com/eclipse-zenoh/zenoh-pico.git --branch 1.1.0 --depth 1
WORKDIR /root/zenoh-pico
RUN /opt/cmake-3.29.6-linux-`arch`/bin/cmake -B build -G Ninja -DFRAG_MAX_SIZE=17000000
RUN ninja -C build && ninja -C build install

# install Zenoh-cpp
WORKDIR /root
RUN git clone https://github.com/eclipse-zenoh/zenoh-cpp.git --branch 1.1.0 --depth 1
WORKDIR /root/zenoh-cpp
RUN /opt/cmake-3.29.6-linux-`arch`/bin/cmake -B build -DZENOHCXX_ZENOHC=off -DZENOHCXX_ZENOHPICO=on
RUN make -C build install

RUN pip install nanobind==2.5.0 numpy==1.26.4
RUN pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

#
#  Optional Dependencies
#
RUN apt update && apt install -y \
    libopencv-dev

RUN pip install pandas ultralytics==8.3.65

#
#  Misc
#

# add a user
RUN useradd ubuntu -m -G sudo -s /usr/bin/bash
RUN passwd -d ubuntu
RUN echo "export PATH=\"/opt/cmake-3.29.6-linux-`arch`/bin:/usr/local/cuda/bin:\${PATH}\"" >> /home/ubuntu/.bashrc
RUN echo 'export PYTHONPATH=./build/src' >> /home/ubuntu/.bashrc

# download YOLO11 models
WORKDIR /home/ubuntu
ADD --chown=ubuntu https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt yolo11n-face.pt
ADD --chown=ubuntu https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt yolo11n-seg.pt
ADD --chown=ubuntu https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt yolo11n-cls.pt

# configure ssh server
RUN mkdir /var/run/sshd
RUN echo 'Port 22' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
