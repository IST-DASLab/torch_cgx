# CGX

TORCH_CGX is a pytorch extension adding a backend for pytorch distributed supporting allreduce of quantized buffers.
It supports quantizations of float16, float32 to 1-8 bits.

TORCH_CGX is based on MPI torch.distributed backend. The extension essentially only replaces allreduce primitive.

## Quick Start

### Prerequisites
TORCH_CGX, as a pytorch extension, requires `pytorch==1.8.0`.

For faster build we recommend to have `ninja` installed (`pip install ninja`).

The compression is only supported for GPU-based buffers so either CUDA or ROCm is required.
If CUDA or ROCm are installed not in the standard paths, set `[CUDA|ROCM]_HOME` or `[CUDA|ROCM]_PATH` accordingly. 

As long as it is based on MPI, it requires OpenMPI with GPU support installed (other MPI implementations were not tested).

### Install
Set `MPI_HOME` environment variable to mpi home. In case of AMD GPU, set `CGX_CUDA` to 0.
Set `CUDA_VECTORIZED` if you want to have compression kernels vectorized (adds ~3% speedup).
```bash
git clone https://github.com/IST-DASLab/pytorch_cgx
export MPI_HOME=/path/to/mpi
python setup.py install
```

### Usage
The only changes to the training script using pytorch distributed required
 are importing the built extension and specifying `cgx` as `torch.distributed.init_process_group` backend parameter.
 
Example:
``` python
import torch
import torch.distributed as dist
import torch_cgx

dist.init_process_group('cgx', init_method='env://', rank=args.local_rank)
```

As long as the extension is based on MPI backend, it requires MPI-compliant launcher (`torch.distributed.launch` won't work):
`$ mpirun -np 2 python train.py`

Also, if your training script was run previously with `torch.distributed.launch` utility, due to MPI launcher you need to set an environment variables (see cifar_train.py in examples)
```
if "OMPI_COMM_WORLD_SIZE" in os.environ:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4040'
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
```

## Tuning
CGX can be tuned with the following environment variables:

- `CGX_COMPRESSION_QUANTIZATION_BITS` - number of bits each value of buffer is quantized to (from 1 to 8). Default is 32 which means no quantization is applied.
- `CGX_COMPRESSION_BUCKET_SIZE` - size of subarray into which buffer is split before quantization. Default is 512.
- `CGX_COMPRESSION_SKIP_INCOMPLETE_BUCKETS` - boolean variable (0 or 1). After the splitting buffer into buckets, some values of buffer may remain. The variable tells quantization algorithm to compress or not to compress the remaining values. Default 0.
- `CGX_COMPRESSION_MINIMAL_SIZE` - minimal size of buffer (number of elements) to compress. Default is 0 but in fact minimal size is forced to be not less than 16.
- `CGX_FUSION_BUFFER_SIZE_MB`. CGX is leveraging [Tensor Fusion](https://github.com/horovod/horovod#tensor-fusion), a performance feature introduced in Horovod. This feature batches small allreduce operations. This decreases a latency in Data Parallel training. The environment variable controls the size of maximal buffer (in MB) that is communicated within one iteration of allreduce algorithm. Default is 64. The variable must be set **before** loading the module.
- `COMMUNICATOR_TYPE`. Specifies what library to use as communication backend (MPI or SHM). SHM shows better performance but is limited to a single node.

## Layer filtering
The extension allows users to separate layers which should be communicated in full-precision.
First, a user needs to register the model layers in the order they would be activated in forward pass:
```
layers = [(name, p.numel()) for name, p in model.named_parameters()]
torch_cgx.register_model(layers)
```
Then, the user needs to exclude the layers or group of layers by their names or part of the names:
```
torch_cgx.exclude_layer("bn") # all batch norm layers
torch_cgx.exclude_layer("bias") # all bias modules
```

IMPORTANT: In the registered model mode `torch_cgx` assumes that all allreduce of buffers with sizes > 16 elements is gradients synchronization.
If during the training user applies torch.distributed.allreduce with such buffers after registering model, it will crash the training.  
## Examples

Basic examples are provided under the [example](example) folder.

## Notes
 - As Compression method, basic stochastic max-min uniform quantization function is used.
 - Reduction algorithm: Scatter-Reduce-AllGather.
 - Part of the source code is based on [Horovod](https://github.com/horovod/horovod) and [NCCL](https://github.com/NVIDIA/nccl) sources.