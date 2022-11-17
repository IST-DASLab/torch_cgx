NUM_NODES=${1:-2}
batch_size=$(( 512 / $NUM_NODES ))

mpirun -np $NUM_NODES --tag-output --allow-run-as-root -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib -mca coll ^hcoll  \
--mca btl_tcp_if_exclude lo,docker0 python cifar_train.py --epochs 10 --dataset-dir ./cifar10 \
--quantization-bits 8 --quantization-bucket-size 1024 --dist-backend cgx --batch-size $batch_size