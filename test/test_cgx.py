import torch.distributed as dist
import torch
import os
import unittest
import warnings
import torch_cgx
import numpy as np

def reduce_equal_tests(rank, world_size, device="cuda"):
    sizes = [1, 2, 8, 128, 1024, 1000000]
    tests = []
    for dtype in [torch.float16, torch.float32, torch.int32]:
        for size in sizes:
            tests.append((
                torch.tensor([rank + 1] * size, dtype=dtype, device=device),
                torch.tensor([(world_size * (world_size + 1)) // 2] * size, dtype=dtype, device=device)
                )
            )
    return tests


def reduce_nonequal_tests(rank, world_size, device="cuda"):
    sizes = [128, 1024, 1025, 16384, 1000000]
    tests = []
    bits = [2, 3, 4, 6, 8]
    for bit in bits:
        for dtype in [torch.float16, torch.float32]:
            for size in sizes:
                arange = np.arange(-size / 2, size / 2, 1.0)
                if dtype == torch.float16:
                    arange *= 1e-3
                tests.append((
                    torch.tensor((rank + 1) * arange, dtype=dtype, device=device),
                    torch.tensor(((world_size * (world_size + 1)) / 2) * arange, dtype=dtype, device=device),
                    bit
                    )
                )
    return tests


class CGXTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(CGXTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def assertTensorEqual(self, t1, t2, msg=None):
        self.assertIsInstance(t1, torch.Tensor, 'First argument is not a Tensor')
        self.assertIsInstance(t2, torch.Tensor, 'Second argument is not a Tensor')
        if not torch.equal(t1, t2):
            self.fail("Tensors are not equal: {} != {}. {}".format(t1, t2, msg))

    def setUp(self) -> None:
        super().setUp()
        self.addTypeEqualityFunc(torch.Tensor, "assertTensorEqual")
        assert "OMPI_COMM_WORLD_SIZE" in os.environ, "Launch with mpirun"
        self.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        self.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '4040'
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        dist.init_process_group(backend="cgx",  init_method="env://", rank=self.rank)
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

    def tearDown(self) -> None:
        dist.barrier()
        dist.destroy_process_group()

    def test_compressed_exact(self):
        quantization_bits = [2, 4, 8]
        tests = reduce_equal_tests(self.rank, self.world_size, device="cuda")
        for q in quantization_bits:
            os.environ["CGX_COMPRESSION_QUANTIZATION_BITS"] = str(q)
            for (input, expected) in tests:
                for i in range(10):
                    t = input.clone()
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    self.assertEqual(t, expected, "Parameters. bits {},buffer size: {}".format(q, t.numel()))

    def test_compressed_non_exact(self):
        tests = reduce_nonequal_tests(self.rank, self.world_size, device="cuda")
        bucket_sizes = [64, 512, 2048]
        for (input, expected, q) in tests:
            os.environ["CGX_COMPRESSION_QUANTIZATION_BITS"] = str(q)
            for bucket_size in bucket_sizes:
                os.environ["CGX_COMPRESSION_BUCKET_SIZE"] = str(bucket_size)
                for i in range(10):
                    t = input.clone()
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    size = t.numel()
                    coef = self.world_size * (self.world_size + 1)
                    self.assertLess(torch.norm(t - expected, p=float("inf")).item(), 2 * min(bucket_size, size) / ((1 << q) - 1) * coef,
                                    "Parameters. bits {}, bucket_size: {}, buffer size: {}".format(q, bucket_size, size))

    def test_uncompressed(self):
        os.environ["CGX_COMPRESSION_QUANTIZATION_BITS"] = str(32)
        tests = reduce_equal_tests(self.rank, self.world_size)
        for (input, expected) in tests:
            t = input.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.assertEqual(t, expected)


if __name__ == "__main__":
    unittest.main()