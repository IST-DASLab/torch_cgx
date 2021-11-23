import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Benchmark settings
parser = argparse.ArgumentParser(description='Imagenet Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--dist-backend', choices=['cgx', 'nccl', 'gloo'], default='nccl',
                    help='Backend for torch distributed')
parser.add_argument('--quantization-bits', type=int, default=32,
                    help='Quantization bits for cgx')
parser.add_argument('--quantization-bucket-size', type=int, default=1024,
                    help='Bucket size for quantization in cgx')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank in distributed launch')


args = parser.parse_args()

if args.dist_backend == "cgx":
    assert "OMPI_COMM_WORLD_SIZE" in os.environ, "Launch with mpirun"
    import torch_cgx
    if 'COMPRESSION_QUANTIZATION_BITS' not in os.environ:
        os.environ['COMPRESSION_QUANTIZATION_BITS'] = str(args.quantization_bits)
    if 'COMPRESSION_BUCKET_SIZE' not in os.environ:
        os.environ['COMPRESSION_BUCKET_SIZE'] = str(args.quantization_bucket_size)


if "OMPI_COMM_WORLD_SIZE" in os.environ:
    args.local_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4040'
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

dist.init_process_group(args.dist_backend, init_method='env://')
args.world_size = dist.get_world_size()

torch.cuda.set_device(args.local_rank)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)
if args.world_size > 1:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
else:
    pass
    # model = torch.nn.DataParallel(model)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
data, target = data.cuda(), target.cuda()


def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


def log(s, nl=True):
    if args.local_rank != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU'
log('Number of %ss: %d' % (device, args.world_size))

if args.dist_backend == 'cgx':
    layers = [(name, p.numel()) for name, p in model.named_parameters()]
    torch_cgx.register_model(layers)
    torch_cgx.exclude_layer("bn")
    torch_cgx.exclude_layer("bias")


# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (args.world_size, device, args.world_size * img_sec_mean, args.world_size * img_sec_conf))