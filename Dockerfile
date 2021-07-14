FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
	git \
	software-properties-common \
	pkg-config \
	unzip \
    wget \
    libgl1-mesa-glx \
    libgl1 \
    zsh \
    python3-pip \
    ninja-build \
    libopenblas-openmp-dev \
    cython

ARG UNAME=christian
ARG UID=10001
ARG GID=100

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

RUN mkdir /data
ARG BASE=/home/christian/workspace/WPI/wpi_gan_generator
RUN mkdir -p ${BASE}

COPY stylegan_code_finder/requirements.txt ${BASE}/requirements.txt

WORKDIR ${BASE}
RUN pip3 install -r requirements.txt

ARG TRAIN_TOOLS_BASE=/opt/train_tools
COPY training_tools ${TRAIN_TOOLS_BASE}/training_tools
RUN cd ${TRAIN_TOOLS_BASE}/training_tools && pip3 install -e .
RUN cd ${BASE}

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
