FROM nvcr.io/nvidia/pytorch:21.09-py3

#RUN apt-get update -qq \
#      && apt-get -y --no-install-recommends install \
#         build-essential \
#         ca-certificates \
#         gdb \
#         gfortran \
#         wget \
#      && apt-get clean all \
#      && rm -r /var/lib/apt/lists/*

#ARG MPI_VERSION="4.1.0"
#ARG MPI_CONFIGURE_OPTIONS="--enable-fast=all,O3 -enable-mpi-cxx --prefix=/usr/openmpi --with-cuda=/usr/local/cuda"
#ARG MPI_MAKE_OPTIONS="-j4"

#RUN mkdir -p /tmp/openmpi-build \
#      && cd /tmp/openmpi-build \
#      && MPI_VER_MM="${MPI_VERSION%.*}" \
#      && wget http://www.openmpi.org/software/ompi/v${MPI_VER_MM}/downloads/openmpi-${MPI_VERSION}.tar.bz2 \
#      && tar xjf openmpi-${MPI_VERSION}.tar.bz2 \
#      && cd openmpi-${MPI_VERSION}  \
#      && ./configure ${MPI_CONFIGURE_OPTIONS} \
#      && make ${MPI_MAKE_OPTIONS} \
#      && make install \
#      && ldconfig \
#      && cd / \
#      && rm -rf /tmp/openmpi-build

#ENV MPI_HOME=/usr/openmpi
#ENV PATH=/usr/openmpi/bin:${PATH}
#ENV LD_LIBRARY_PATH=/usr/openmpi/lib:${LD_LIBRARY_PATH}
ENV MPI_HOME=/opt/hpcx/ompi/

COPY . /torch_qmpi

RUN cd /torch_qmpi && python setup.py install