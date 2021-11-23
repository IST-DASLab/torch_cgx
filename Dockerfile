FROM nvcr.io/nvidia/pytorch:21.09-py3

ENV MPI_HOME=/opt/hpcx/ompi/

RUN git clone https://github.com/IST-DASLab/torch_cgx --branch torch/1.9 --single-branch /torch_cgx &&\
cd /torch_cgx && \
python setup.py install