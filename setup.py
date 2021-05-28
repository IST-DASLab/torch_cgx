from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

MPI_HOME=os.environ.get("MPI_HOME", "/usr/local/mpi")
IS_CUDA=int(os.environ.get("QMPI_CUDA", "1")) != 0
src = []
root="src"
exclude_prefix = "hip_"
if not IS_CUDA:
    exclude_prefix = "cuda_"

for path, dirs, files in os.walk(root):
    for file in files:
        if (".cc" in file or '.cu' in file):
            if exclude_prefix not in file and "impl" not in file:
                src.append(os.path.join(path, file))


compile_args = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
if IS_CUDA:
    compile_args.append("-DHAVE_CUDA=1")
else:
    compile_args.append("-DHAVE_ROCM=1")
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5+PTX"

setup(name='torch_qmpi',
      ext_modules=[cpp_extension.CUDAExtension('torch_qmpi', sources=src,
                                              include_dirs=[os.path.join(MPI_HOME, "include")],
                                              extra_compile_args=compile_args,
                                              extra_link_args=['-L'+ os.path.join(MPI_HOME, '/lib'),
                                                               '-lmpi_cxx', '-lmpi'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      )


