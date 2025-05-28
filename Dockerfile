ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# create input/output directory
RUN mkdir /input /output && \
    chown user:user /input /output

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam
USER root

# Set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# install libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    curl \
    openjdk-17-jre \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python
RUN apt-get update && apt-get install -y python3-pip python3-dev python-is-python3

# Switch to user
USER user
WORKDIR /opt/app/

# install python libraries
RUN python -m pip install --upgrade pip setuptools pip-tools
COPY --chown=user:user requirements.in .
RUN python -m pip install --no-cache-dir -r requirements.in

# install unicorn_eval
COPY --chown=user:user . /opt/app/unicorn_eval
RUN python -m pip install /opt/app/unicorn_eval

# download Bert model weights
RUN mkdir -p /opt/app/unicorn_eval/models/dragon-bert-base-mixed-domain && \
    python -c "\
from transformers import AutoModel, AutoTokenizer; \
model = AutoModel.from_pretrained('joeranbosma/dragon-bert-base-mixed-domain'); \
tokenizer = AutoTokenizer.from_pretrained('joeranbosma/dragon-bert-base-mixed-domain'); \
model.save_pretrained('/opt/app/unicorn_eval/models/dragon-bert-base-mixed-domain'); \
tokenizer.save_pretrained('/opt/app/unicorn_eval/models/dragon-bert-base-mixed-domain')"


ENTRYPOINT ["unicorn_eval"]
