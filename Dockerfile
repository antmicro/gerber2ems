ARG BASE_IMAGE=debian:trixie

FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

RUN apt update \
    && apt -qqy install \
      # OpenEMS
      build-essential cmake git libhdf5-dev libvtk9-dev libboost-all-dev libcgal-dev libtinyxml-dev qtbase5-dev libvtk9-qt-dev python3 python3-venv cython3 pip \
      # gerber2ems
      gerbv \
    && rm -rf /var/lib/apt/lists/*

RUN useradd docker && echo "docker:docker" | chpasswd && mkdir /home/docker && chown -R docker:docker /home/docker

USER docker

ENV HOME="/home/docker" \
    XDG_CONFIG_HOME="/home/docker/.config" \
    XDG_DATA_HOME="/home/docker/.local/share" \
    XDG_BIN_HOME="/home/docker/.local/bin" \
    PATH="/home/docker/.local/bin:$PATH"

WORKDIR /home/docker

RUN echo "Installing openEMS..." \
  && git clone https://github.com/thliebig/openEMS-Project.git \
  && pushd ./openEMS-Project \
  && git checkout a30587728affa4f8451e13819981899bd8ab6b64 \
  && git submodule update --init --recursive \
  && ./update_openEMS.sh ~/opt/openEMS --python \
  && popd

COPY --chown=docker:docker . gerber2ems

RUN echo "Installing gerber2ems..." \
  && pushd ./gerber2ems \
  && source ~/opt/openEMS/venv/bin/activate \
  && pip install . \
  && mkdir --parents ~/.local/bin \
  && ln -s ~/opt/openEMS/venv/bin/{gerber2ems,ems2paraview,ems2png} ~/.local/bin \
  && popd

ENTRYPOINT ["gerber2ems"]
