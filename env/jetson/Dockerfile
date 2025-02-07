# see README.md to learn how to build this image
FROM torchvision:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

#
#  Tools
#
RUN apt update && apt install -y \
    build-essential ninja-build \
    sudo curl wget git tar tmux vim openssh-server

# install CMake
WORKDIR /opt
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6-linux-`arch`.tar.gz"
RUN tar xvf cmake-3.29.6-linux-`arch`.tar.gz

#
#  Dependencies
#

# install prerequisites of Iceoryx
RUN apt install -y gcc g++ cmake libacl1-dev libncurses5-dev pkg-config

# install Iceoryx
WORKDIR /root
RUN git clone https://github.com/eclipse-iceoryx/iceoryx.git --branch v2.0.6 --depth 1
WORKDIR /root/iceoryx
RUN /opt/cmake-3.29.6-linux-`arch`/bin/cmake -B build -G Ninja iceoryx_meta
RUN ninja -C build && ninja -C build install

RUN pip install --upgrade pip
RUN pip install nanobind==2.5.0 numpy==1.26.4

#
#  Optional Dependencies
#
RUN apt update && apt install -y \
    libopencv-dev

RUN pip install pandas ultralytics==8.3.65

# install dependencies required by ultralytics
RUN apt install -y \
    libhdf5-dev \
    libtesseract-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libatlas-base-dev \
    libtbb-dev

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
ADD --chown ubuntu https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt yolo11n-face.pt
ADD --chown ubuntu https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt yolo11n-seg.pt

# configure ssh server
RUN mkdir /var/run/sshd
RUN echo 'Port 22' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'PermitEmptyPasswords yes' >> /etc/ssh/sshd_config
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
