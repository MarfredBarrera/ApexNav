# ApexNav on ROS 2 Jazzy - headless, GPU, Foxglove instead of RViz.
#
# Two Python environments live in this image, on purpose:
#
#   Env A  /opt/venv/apexnav          python 3.12  ROS 2 + Habitat + your code
#   Env B  /opt/conda/envs/apexnav_vlm python 3.9  the four VLM servers
#
# Jazzy's rclpy is a cpython-312 extension and every published habitat-sim conda
# build is py3.9-only, so the two cannot share an interpreter with binary packages.
# Env A resolves that by building habitat-sim v0.3.4 from source on 3.12 (the first
# release whose build matrix targets 3.10-3.12). Env B keeps the README's py3.9 pins
# because salesforce-lavis has no cp312 story. They talk over localhost HTTP, which
# is why they share one container. See docs/DOCKER_JAZZY_FOXGLOVE.md.

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    EGL_PLATFORM=surfaceless

# ---------------------------------------------------------------------------
# 1. ROS 2 Jazzy
# ---------------------------------------------------------------------------
# ros-base rather than ros-desktop: RViz2 and its Qt stack are dead weight here
# because foxglove_bridge replaces them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 locales tzdata \
 && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-base \
        ros-jazzy-cv-bridge \
        ros-jazzy-pcl-conversions \
        ros-jazzy-pcl-ros \
        ros-jazzy-message-filters \
        ros-jazzy-tf2 \
        ros-jazzy-tf2-ros \
        ros-jazzy-tf2-eigen \
        ros-jazzy-tf2-geometry-msgs \
        ros-jazzy-tf-transformations \
        ros-jazzy-visualization-msgs \
        ros-jazzy-nav-msgs \
        ros-jazzy-sensor-msgs \
        ros-jazzy-rosidl-default-generators \
        ros-jazzy-rosidl-default-runtime \
        ros-jazzy-foxglove-bridge \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Native build dependencies
# ---------------------------------------------------------------------------
# libarmadillo-dev + libompl-dev are the README's system deps; the mesa/x11/glm/
# jpeg/bullet set is what habitat-sim needs to compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git wget unzip zip pkg-config \
        libeigen3-dev libpcl-dev libarmadillo-dev libompl-dev \
        libjpeg-dev libglm-dev libegl1-mesa-dev libgl1-mesa-dev \
        xorg-dev freeglut3-dev libhdf5-dev libbullet-dev \
        python3-venv python3-dev python3-pip \
        tmux vim less htop iproute2 iputils-ping \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. OSQP + osqp-eigen (required by trajectory_manager)
# ---------------------------------------------------------------------------
RUN git clone --recursive -b v0.6.3 https://github.com/osqp/osqp.git /tmp/osqp \
 && cmake -S /tmp/osqp -B /tmp/osqp/build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /tmp/osqp/build -j"$(nproc)" \
 && cmake --install /tmp/osqp/build \
 && git clone -b v0.8.1 https://github.com/robotology/osqp-eigen.git /tmp/osqp-eigen \
 && cmake -S /tmp/osqp-eigen -B /tmp/osqp-eigen/build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /tmp/osqp-eigen/build -j"$(nproc)" \
 && cmake --install /tmp/osqp-eigen/build \
 && ldconfig \
 && rm -rf /tmp/osqp /tmp/osqp-eigen

# ---------------------------------------------------------------------------
# 4. Env A - Python 3.12 venv
# ---------------------------------------------------------------------------
# --system-site-packages is load-bearing: it is how /opt/ros/jazzy's rclpy,
# cv_bridge and generated message modules stay importable from inside the venv.
ENV APEX_CORE_VENV=/opt/venv/apexnav
RUN python3 -m venv --system-site-packages ${APEX_CORE_VENV} \
 && ${APEX_CORE_VENV}/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# CUDA torch, not the CPU build: the VLM *clients* import torch/torchvision at module
# level, and having CUDA here means new perception modules can run inline in the
# habitat process instead of being forced behind an HTTP server.
#
# NOT the README's torch 2.5.0/cu124: that pins nvidia-cudnn-cu12==9.1.0.70, which has
# since been pulled from the index (available versions jump 9.0.0.312 -> 9.1.1.17), so
# the pin no longer resolves at all. 2.8.0/0.23.0 on cu126 is the newest pair with both
# cp312 and cp39 wheels, so Env A and Env B can stay on one version, and cu126 matches
# the base image's CUDA. Keep both envs in sync if you bump this.
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN ${APEX_CORE_VENV}/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX}

COPY requirements-core.txt /tmp/requirements-core.txt
RUN ${APEX_CORE_VENV}/bin/pip install --no-cache-dir -r /tmp/requirements-core.txt

# ---------------------------------------------------------------------------
# 5. habitat-sim from source (the long one, ~30 min)
# ---------------------------------------------------------------------------
# v0.3.4 moved to scikit-build-core, so the README's and issue #36's
# `python setup.py install --headless --bullet` no longer works - setup.py is now a
# shim that exits 1. Options are env vars instead; Bullet is ON by default.
# --no-deps drops its numpy>=2.0 requirement so habitat-lab's numpy==1.26.4 wins,
# and --no-build-isolation makes it compile against that same numpy.
ARG HABITAT_SIM_TAG=v0.3.4
RUN git clone --branch ${HABITAT_SIM_TAG} --depth 1 --recursive \
        https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim \
 && cd /opt/habitat-sim \
 && HABITAT_BUILD_GUI_VIEWERS=OFF \
    ${APEX_CORE_VENV}/bin/pip install --no-cache-dir --no-build-isolation --no-deps . \
 && rm -rf /opt/habitat-sim/build /opt/habitat-sim/.git

# ---------------------------------------------------------------------------
# 6. habitat-lab
# ---------------------------------------------------------------------------
ARG HABITAT_LAB_TAG=v0.3.4
RUN git clone --branch ${HABITAT_LAB_TAG} --depth 1 \
        https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab \
 && ${APEX_CORE_VENV}/bin/pip install --no-cache-dir -e /opt/habitat-lab/habitat-lab \
 && ${APEX_CORE_VENV}/bin/pip install --no-cache-dir -e /opt/habitat-lab/habitat-baselines

# ---------------------------------------------------------------------------
# 7. Env B - conda python 3.9 for the VLM servers
# ---------------------------------------------------------------------------
# Miniforge, not the README's Miniconda: Miniconda's `defaults` channels now refuse
# non-interactive use until Anaconda's Terms of Service are accepted
# (CondaToSNonInteractiveError), and those terms carry commercial-use conditions.
# Miniforge is the same conda, defaulting to conda-forge, with neither problem.
# Only used to get a python 3.9 interpreter - every package here comes from pip.
ENV CONDA_DIR=/opt/conda \
    APEX_VLM_ENV=/opt/conda/envs/apexnav_vlm
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
        -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
 && rm /tmp/miniforge.sh \
 && ${CONDA_DIR}/bin/conda create -y -n apexnav_vlm python=3.9 \
 && ${CONDA_DIR}/bin/conda clean -afy

RUN ${APEX_VLM_ENV}/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX}

COPY requirements-vlm.txt /tmp/requirements-vlm.txt
RUN ${APEX_VLM_ENV}/bin/pip install --no-cache-dir -r /tmp/requirements-vlm.txt

# lavis last, and with --no-deps: its declared deps include an *unpinned* spacy, which
# now resolves to a version needing thinc>=8.3.12 / Python>=3.10 and makes the whole
# env unsolvable. requirements-vlm.txt carries the subset its import path actually uses,
# including a py3.9-compatible spacy pin. Installing it after the pins also keeps it
# from disturbing transformers.
RUN ${APEX_VLM_ENV}/bin/pip install --no-cache-dir --no-deps salesforce-lavis==1.0.2

# ---------------------------------------------------------------------------
# 8. External code checkouts (README section 1.3)
# ---------------------------------------------------------------------------
# Both are referenced by CWD-relative paths in the source
# (sys.path.insert(0, "yolov7/") and "GroundingDINO/groundingdino/config/..."),
# so the entrypoint symlinks them into the workspace at container start.
RUN git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git \
        /opt/external/GroundingDINO \
 && git clone --depth 1 https://github.com/WongKinYiu/yolov7.git \
        /opt/external/yolov7

# torch 2.6 flipped torch.load's default to weights_only=True, so yolov7's checkpoint
# load dies with "Weights only load failed ... models.yolo.Model". yolov7 predates the
# change and passes no flag. The README's torch 2.5 would not hit this, but that pin is
# no longer installable (see the torch section), so patch the loader instead.
# yolov7-e6e.pt is the official release artifact, so unpickling it is not a new risk.
RUN sed -i 's/torch\.load(w, map_location=map_location)/torch.load(w, map_location=map_location, weights_only=False)/' \
        /opt/external/yolov7/models/experimental.py \
 && grep -q "weights_only=False" /opt/external/yolov7/models/experimental.py

# Tensor::type() no longer works with current torch. The ros2-jazzy README suggests
# replacing value.type() with value.scalar_type(), but that is only correct for one of
# the two uses - a blanket substitution produces
#   error: expression must have class type but it has type "c10::ScalarType"
# because it also rewrites the AT_ASSERTM device checks. The two cases need different
# fixes, and the order matters:
#   X.type().is_cuda()               -> X.is_cuda()          (11 sites, 6 tensors)
#   AT_DISPATCH_FLOATING_TYPES(X.type(), ...) -> X.scalar_type()  (2 sites)
RUN find /opt/external/GroundingDINO/groundingdino/models/GroundingDINO/csrc \
        -type f \( -name '*.cu' -o -name '*.cpp' -o -name '*.h' \) -print0 \
    | xargs -0 sed -i \
        -e 's/\([A-Za-z_][A-Za-z0-9_]*\)\.type()\.is_cuda()/\1.is_cuda()/g' \
        -e 's/\([A-Za-z_][A-Za-z0-9_]*\)\.type()/\1.scalar_type()/g'

# TORCH_CUDA_ARCH_LIST=8.9 targets the L40S only, which cuts this build from
# ~20 min to ~3. Change it if you move to a different GPU.
RUN cd /opt/external/GroundingDINO \
 && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.9" \
    ${APEX_VLM_ENV}/bin/pip install --no-cache-dir --no-build-isolation -e .

# Env A only needs groundingdino.datasets.transforms (imported unguarded by
# vlm/detector/grounding_dino.py); that path never touches the _C extension, so a
# .pth pointing at the same checkout is enough - no second compile.
RUN echo "/opt/external/GroundingDINO" \
        > ${APEX_CORE_VENV}/lib/python3.12/site-packages/groundingdino_src.pth

# ---------------------------------------------------------------------------
# 9. ollama (optional LLM path; llm/answers ships pre-generated output)
# ---------------------------------------------------------------------------
# Models land on the mounted data volume so the ~5 GB qwen3:8b stays out of the image.
# zstd is installed here rather than with the other apt packages so this stays a late,
# cheap layer - the installer needs it to unpack, and adding it earlier would invalidate
# the torch and habitat-sim caches. The installer's systemd warning is harmless in Docker.
RUN apt-get update && apt-get install -y --no-install-recommends zstd \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL https://ollama.com/install.sh | sh
ENV OLLAMA_MODELS=/workspace/ApexNav/data/ollama

# ---------------------------------------------------------------------------
# 10. User + shell environment
# ---------------------------------------------------------------------------
# Runs as container root by default, which is the correct choice under *rootless*
# Docker (what this host uses): rootless maps container uid 0 to your host uid, so
# bind-mounted writes - colcon's build/ install/ log/, videos/, outputs/ - land owned
# by you, and container root has no privilege on the host. A non-root container user
# would instead map into the subuid range and get EACCES on your own files.
#
# Under *rootful* Docker the mapping is identity and container root would write
# root-owned files into your checkout. There, set an explicit user in compose:
#     user: "1018:1020"
# The apex account below exists so that override has something to select.
ARG UID=1018
ARG GID=1020
RUN (getent group ${GID} || groupadd -g ${GID} apex) \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash apex \
 && mkdir -p /workspace/ApexNav

COPY apexnav-env.sh /opt/apexnav/apexnav-env.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /opt/apexnav/apexnav-env.sh \
 && echo 'source /opt/apexnav/apexnav-env.sh' >> /root/.bashrc \
 && echo 'source /opt/apexnav/apexnav-env.sh' >> /home/apex/.bashrc

WORKDIR /workspace/ApexNav

# The repo is used from the source tree rather than pip-installed: every data path
# in it is CWD-relative anyway, and this keeps edits live in both environments with
# no reinstall step.
ENV PYTHONPATH=/workspace/ApexNav

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
