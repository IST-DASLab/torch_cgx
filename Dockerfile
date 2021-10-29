FROM nvcr.io/nvidia/pytorch:21.09-py3

ARG GITHUB_LOGIN
ARG GITHUB_TOKEN
ENV MPI_HOME=/opt/hpcx/ompi/

RUN git clone https://${GITHUB_LOGIN}:${GITHUB_TOKEN}@github.com/IST-DASLab/torch_qmpi --branch torch/1.9 --single-branch /torch_qmpi &&\
cd /torch_qmpi && \
python setup.py install