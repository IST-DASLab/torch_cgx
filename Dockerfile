FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

ENV MPI_HOME=/opt/hpcx/ompi/
ENV NCCL_INCLUDE=/usr/include
ENV NCCL_LIB=/usr/lib/x86_64-linux-gnu/

RUN git clone https://github.com/IST-DASLab/torch_cgx /torch_cgx && \
cd /torch_cgx && python setup.py install