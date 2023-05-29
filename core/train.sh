OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='1','2' python -m torch.distributed.launch --nproc_per_node=2 --master_port 10000  main.py \

