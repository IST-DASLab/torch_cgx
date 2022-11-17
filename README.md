# CGX

CGX is a pytorch extension adding a backend for pytorch distributed supporting allreduce of quantized buffers.
It supports quantizations of float16, float32 to 1-8 bits.

CGX is based on MPI torch.distributed backend. The extension essentially only replaces allreduce primitive.

## Quick Start

### Prerequisites
CGX, as a pytorch extension, requires `pytorch>=1.10.0`.

For faster build we recommend to have `ninja` installed (`pip install ninja`).

The compression is only supported for GPU-based buffers so either CUDA or ROCm is required.
If CUDA or ROCm are installed not in the standard paths, set `[CUDA|ROCM]_HOME` or `[CUDA|ROCM]_PATH` accordingly. 

As long as it is based on MPI, it requires OpenMPI with GPU support installed (other MPI implementations were not tested).
Also, the library supports NCCL based communications, so it requires NVIDIA NCCL library.
### Install
```bash
export MPI_HOME=/path/to/mpi
export NCCL_HOME=/path/to/nccl
pip install pytorch-cgx
```

### Build from source
Set `MPI_HOME` environment variable to mpi home. In case of AMD GPU, set `CGX_CUDA` to 0.
Set `NCCL_HOME` environment variable to NCCL home, or `NCCL_INCLUDE` and `NCCL_LIB`.
Set `QSGD_DETERMENISTIC=0` if you want to have stochastic version QSGD.

```bash
git clone https://github.com/IST-DASLab/torch_cgx
cd torch_cgx
export MPI_HOME=/path/to/mpi
export NCCL_HOME=/path/to/nccl
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
Also, it order to perform layerwise compression and being able to filter small sensitive to gradient compression
layers (typically these are batch norm layers and biases) the `cgx` needs to have information about the model.
For that users need to register the communication hook. The minimal size of the layers can be 
controlled with `layer_min_size` parameter.

``` python
model = torch.nn.parallel.DistributedDataParallel(...)
from cgx_utils import cgx_hook, CGXState
state = CGXState(torch.distributed.group.WORLD, layer_min_size=1024,
                  compression_params={"bits": args.quantization_bits,
                                      "bucket_size": args.quantization_bucket_size})
model.register_comm_hook(state, cgx_hook)
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

- `CGX_COMPRESSION_QUANTIZATION_BITS` - number of bits each value of buffer is quantized to (from 1 to 8). Default is 32 which means no quantization is applied. This variable must be used if the `cgx_hook` communication hook is not registered.
- `CGX_COMPRESSION_BUCKET_SIZE` - size of subarray into which buffer is split before quantization. Default is 512.
- `CGX_COMPRESSION_SKIP_INCOMPLETE_BUCKETS` - boolean variable (0 or 1). After the splitting buffer into buckets, some values of buffer may remain. The variable tells quantization algorithm to compress or not to compress the remaining values. Default 0.
- `CGX_COMPRESSION_MINIMAL_SIZE` - minimal size of buffer (number of elements) to compress. Default is 0 but in fact minimal size is forced to be not less than 16.
- `CGX_FUSION_BUFFER_SIZE_MB`. CGX is leveraging [Tensor Fusion](https://github.com/horovod/horovod#tensor-fusion), a performance feature introduced in Horovod. This feature batches small allreduce operations. This decreases a latency in Data Parallel training. The environment variable controls the size of maximal buffer (in MB) that is communicated within one iteration of allreduce algorithm. Default is 64. The variable must be set **before** loading the module.
- `CGX_INNER_COMMUNICATOR_TYPE`. Specifies what library to use as communication backend for intra node communication (MPI, SHM, NCCL).
- `CGX_CROSS_COMMUNICATOR_TYPE`. Specifies what library to use as communication backend for inter node communication (MPI, NCCL).
- `CGX_INTRA_BROADCAST`. Parameter for multinode training. When enabled, inter-node communication is performed by only one gpu per node.

## Examples

Basic examples are provided under the [example](examples) folder.

## Notes
 - As Compression method, basic max-min uniform quantization function is used. In order to use max-min with random rounding like in QSGD, compile the library with QSGD_DETERMINISTIC=0
 - Reduction algorithm: Scatter-Reduce-AllGather.
 - Part of the source code is based on [Horovod](https://github.com/horovod/horovod) and [NCCL](https://github.com/NVIDIA/nccl) sources.