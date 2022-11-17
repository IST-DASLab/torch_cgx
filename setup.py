from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
import subprocess
from pathlib import Path

src = ['src/mpi_allreduce_operations.cc', 'src/ProcessGroupCGX.cc',
       'src/common/reducer.cc', 'src/common/buffer.cc', 'src/common/mpi_context.cc',
       'src/common/mpi_communicator.cc', 'src/common/shm_communicator.cc',
       'src/common/scatter_reduce_allgather.cc', 'src/common/ring.cc', 'src/common/utils.cc',
       'src/common/compressor.cc', 'src/common/layer.cc', 'src/common/shm_utils.cc',
       'src/common/compression/gpu_compression_operations.cc', 'src/common/nccl_reduce.cc']

MPI_HOME=os.environ.get("MPI_HOME", "/usr/local/mpi")
NCCL_HOME=os.environ.get("NCCL_HOME", "/usr/local/nccl")
NCCL_INCLUDE=os.environ.get("NCCL_INCLUDE", "/usr/local/nccl/include")
NCCL_LIB=os.environ.get("NCCL_LIB", "/usr/local/nccl/lib")
IS_CUDA=int(os.environ.get("CGX_CUDA", "1")) != 0
CUDA_VECTORIZED=int(os.environ.get("CUDA_VECTORIZED", "1")) != 0
QSGD_DETERMENISTIC=int(os.environ.get("QSGD_DETERMENISTIC", "1")) != 0
link_args = ['-L'+ os.path.join(MPI_HOME, 'lib'), '-lmpi']
ompi_info_bin=os.path.join(MPI_HOME, 'bin', 'ompi_info')
env = os.environ
try:
    ompi_info_out = subprocess.check_output([ompi_info_bin, '--parsable'], env=env)
except OSError as e:
    raise RuntimeError('CMake failed: {}'.format(str(e)))

if "bindings:cxx:yes" in str(ompi_info_out):
    link_args.append('-lmpi_cxx')
if IS_CUDA:
    src.extend(['src/common/compression/cuda_compression_operations.cu', 'src/common/cuda_operations.cc'])
else:
    src.extend(['src/common/compression/hip_compression_operations.cc', 'src/common/hip_operations.cc'])
include_dirs = [os.path.join(MPI_HOME, "include")]
cxx_compile_args = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
nvcc_compile_args = []
if IS_CUDA:
    cxx_compile_args.append("-DHAVE_CUDA=1")
    nvcc_compile_args.append("-DHAVE_CUDA=1")
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.6'
else:
    cxx_compile_args.append("-DHAVE_ROCM=1")
    nvcc_compile_args.append("-DHAVE_ROCM=1")

if os.path.isdir(NCCL_HOME):
    include_dirs.append(os.path.join(NCCL_HOME, "include"))
    link_args.append('-L' + os.path.join(NCCL_HOME, 'lib'))
    link_args.append('-lnccl')
elif os.path.isdir(NCCL_INCLUDE) and os.path.isdir(NCCL_LIB):
    include_dirs.append(os.path.join(NCCL_INCLUDE))
    link_args.append('-L' + os.path.join(NCCL_LIB))
    link_args.append('-lnccl')
else:
    raise ValueError("NCCL is not available")

if CUDA_VECTORIZED:
    nvcc_compile_args.append("-DCUDA_VECTORIZED=1")
    cxx_compile_args.append("-DCUDA_VECTORIZED=1")
if QSGD_DETERMENISTIC:
    nvcc_compile_args.append("-DQSGD_DETERMENISTIC=1")
    cxx_compile_args.append("-DQSGD_DETERMENISTIC=1")

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='pytorch_cgx',
      packages=['cgx_utils'],
      version='0.1.0',
      description='pytorch extension adding a backend '
                  'supporting allreduce of quantized buffers.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Ilia Markov',
      author_email='ilia.markov@ist.ac.at',
      url='https://github.com/IST-DASLab/torch_cgx/',
      download_url="https://github.com/IST-DASLab/torch_cgx/archive/refs/tags/v0.1.0.tar.gz",
      ext_modules=[cpp_extension.CUDAExtension('torch_cgx', sources=src,
                                              include_dirs=include_dirs,
                                              extra_compile_args={'cxx': cxx_compile_args, 'nvcc': nvcc_compile_args},
                                              extra_link_args=link_args)],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )
