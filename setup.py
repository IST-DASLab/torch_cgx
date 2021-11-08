from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# src = []
# root="src"
# exclude_prefix = "hip_"
# if not IS_CUDA:
#     exclude_prefix = "cuda_"

src = ['src/mpi_allreduce_operations.cc', 'src/qmpi.cc',
       'src/common/reducer.cc', 'src/common/buffer.cc', 'src/common/mpi_context.cc',
       'src/common/mpi_communicator.cc', 'src/common/p2p_communicator.cc', 'src/common/shm_communicator.cc',
       'src/common/scatter_reduce_allgather.cc', 'src/common/ring.cc', 'src/common/utils.cc',
       'src/common/compressor.cc', 'src/common/layer.cc', 'src/common/shm_utils.cc',
       'src/common/compression/gpu_compression_operations.cc']

# for path, dirs, files in os.walk(root):
#     for file in files:
#         if (".cc" in file or '.cu' in file):
#             if exclude_prefix not in file and "impl" not in file:
#                 src.append(os.path.join(path, file))
MPI_HOME=os.environ.get("MPI_HOME", "/usr/local/mpi")
IS_CUDA=int(os.environ.get("QMPI_CUDA", "1")) != 0
CUDA_VECTORIZED=int(os.environ.get("CUDA_VECTORIZED", "1")) != 0

if IS_CUDA:
    src.extend(['src/common/compression/cuda_compression_operations.cu', 'src/common/cuda_operations.cc'])
else:
    src.extend(['src/common/compression/hip_compression_operations.cc', 'src/common/hip_operations.cc'])

cxx_compile_args = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
nvcc_compile_args = []
if IS_CUDA:
    cxx_compile_args.append("-DHAVE_CUDA=1")
    nvcc_compile_args.append("-DHAVE_CUDA=1")
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.6'
else:
    cxx_compile_args.append("-DHAVE_ROCM=1")
    nvcc_compile_args.append("-DHAVE_ROCM=1")

if CUDA_VECTORIZED:
    nvcc_compile_args.append("-DCUDA_VECTORIZED=1")

setup(name='torch_qmpi',
      version='0.0.1',
      ext_modules=[cpp_extension.CUDAExtension('torch_qmpi', sources=src,
                                              include_dirs=[os.path.join(MPI_HOME, "include")],
                                              extra_compile_args={'cxx': cxx_compile_args, 'nvcc': nvcc_compile_args},
                                              extra_link_args=['-L'+ os.path.join(MPI_HOME, 'lib'),
                                                               '-lmpi'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )
