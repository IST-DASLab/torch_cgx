NUM_NODES=${1:-2}
batch_size=$(( 512 / $NUM_NODES ))
mpirun -np $NUM_NODES -mca pml ob1 python examples/cifar_train.py --epochs 10 --dataset-dir ~/Datasets/cifar10 \
--quantization-bits 4 --quantization-bucket-size 1024 --dist-backend qmpi --batch-size $batch_size