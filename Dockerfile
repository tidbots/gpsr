# Dockerfile
FROM ros:noetic-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# === ユーザ設定 ===
ARG USER_NAME=roboworks
ARG GROUP_NAME=roboworks
ARG UID=1000
ARG GID=1000

# ユーザ作成
RUN groupadd -g ${GID} ${GROUP_NAME} \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER_NAME} \
 && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# === 必要パッケージ ===
RUN apt-get update && apt-get install -y \
    ros-noetic-audio-common \
    build-essential \
    ros-noetic-catkin \
    git \
    python3-pip \
    python3-numpy \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libasound2-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir torch torchaudio
 RUN pip3 install --no-cache-dir --upgrade pip \
 && pip3 install --no-cache-dir torch torchaudio packaging
 
# === catkin workspace ===
ENV CATKIN_WS=/hsr_ws
RUN mkdir -p ${CATKIN_WS}/src
WORKDIR ${CATKIN_WS}

# Copy user src/ directory
COPY src ${CATKIN_WS}/src

# === catkin build ===
RUN . /opt/ros/noetic/setup.sh \
 && catkin_make

# === setup.bash を自動ロード ===
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/${USER_NAME}/.bashrc \
 && echo "source /hsr_ws/devel/setup.bash" >> /home/${USER_NAME}/.bashrc

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
