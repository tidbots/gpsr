# Dockerfile
FROM ros:noetic-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# === ユーザ設定（必要に応じて変更） ===
ARG USER_NAME=roboworks
ARG GROUP_NAME=roboworks
ARG UID=1000
ARG GID=1000

# ユーザ作成（.ros PermissionError 回避）
RUN groupadd -g ${GID} ${GROUP_NAME} \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USER_NAME} \
 && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# === 依存パッケージインストール ===
RUN apt-get update && apt-get install -y \
    # audio_capture / audio_common
    ros-noetic-audio-common ros-noetic-smach\
    # ビルド・ユーティリティ
    build-essential \
    ros-noetic-catkin \
    git \
    # Python / numpy
    python3-pip \
    python3-numpy \
    # オーディオ関係ツール
    alsa-utils \
    pulseaudio \
    pulseaudio-utils \
    libasound2-dev \
 && rm -rf /var/lib/apt/lists/*

# pip は後で Whisper / VAD 用にも使える
RUN pip3 install --no-cache-dir --upgrade pip

# === catkin ワークスペース作成 ===
ENV CATKIN_WS=/hsr_ws
RUN mkdir -p ${CATKIN_WS}/src
WORKDIR ${CATKIN_WS}

# hsr_audio_pipeline パッケージをコピー
# （repo_root/hsr_audio_pipeline → /hsr_ws/src/hsr_audio_pipeline）
COPY src/hsr_audio_pipeline ${CATKIN_WS}/src/hsr_audio_pipeline

# === ワークスペースビルド ===
RUN . /opt/ros/noetic/setup.sh \
 && catkin_make

# === シェル起動時の環境設定 ===
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/${USER_NAME}/.bashrc \
 && echo "source /hsr_ws/devel/setup.bash" >> /home/${USER_NAME}/.bashrc

USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# ベースイメージのエントリポイントを利用
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
